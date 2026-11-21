"""Data sources for UroSentinel.

Four kinds of data are supported, all mapped to one harmonised (shared) column layout so the
same cleaning, fairness and modelling code works on each:

1. NHANES (United States): the National Health and Nutrition Examination Survey, a nationally
   representative survey. Public, downloaded directly from the CDC website for several cycles.
2. UCI Acute Inflammations (international, Poland): a small public clinical dataset of urinary
   bladder inflammation and kidney (renal pelvis) nephritis, originally built to test a rough
   set expert system.
3. UK data: UK Biobank, CPRD, OpenSAFELY or a local NHS audit. These need approved access, so
   nothing is downloaded. A loader checks a CSV you supply against the harmonised layout.
4. Clinical differential cohort: the six-condition layout used by "Urology prediction model 2".
   Either your own CSV, or a clearly labelled synthetic cohort for demonstration.

Every loader records where its data came from in ``df.attrs['provenance']`` and whether it is
synthetic in ``df.attrs['synthetic']``. The ethics module reads these into the model card.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# NHANES
# ---------------------------------------------------------------------------------------------

# Cycle label -> (file suffix, first year). The 2019-2020 cycle was cut short by COVID-19 and is
# released only in a merged 2017-March 2020 form, so it is not listed separately.
NHANES_CYCLES = {
    "2005-2006": ("D", 2005),
    "2007-2008": ("E", 2007),
    "2009-2010": ("F", 2009),
    "2011-2012": ("G", 2011),
    "2013-2014": ("H", 2013),
    "2015-2016": ("I", 2015),
    "2017-2018": ("J", 2017),
    "2021-2023": ("L", 2021),
}

# The files used from each cycle: demographics, kidney and urology questions, diabetes, body measures.
NHANES_FILES = ["DEMO", "KIQ_U", "DIQ", "BMX"]

# NHANES codes for "refused" and "don't know". They are not real answers, so they become missing.
NHANES_NONRESPONSE = {7, 9, 77, 99, 777, 999, 7777, 9999}

ETHNICITY_LABELS = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                    4: "Non-Hispanic Black", 5: "Other or multiracial"}


def _nhanes_urls(file: str, cycle: str) -> list[str]:
    """Return candidate download addresses for one NHANES file (newest CDC layout first)."""
    suffix, year = NHANES_CYCLES[cycle]
    name = f"{file}_{suffix}"
    return [
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt",
        f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{name}.XPT",
    ]


def _read_xpt_bytes(raw: bytes) -> pd.DataFrame:
    df = pd.read_sas(io.BytesIO(raw), format="xport")
    # SAS transport files store zero as a tiny number (about 5e-79). Round those back to 0.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    return df


def download_nhanes_file(file: str, cycle: str, cache_dir: str = "nhanes_cache") -> pd.DataFrame | None:
    """Download (or read from cache) one NHANES file. Returns None if it cannot be fetched."""
    os.makedirs(cache_dir, exist_ok=True)
    suffix, _ = NHANES_CYCLES[cycle]
    path = os.path.join(cache_dir, f"{file}_{suffix}.xpt")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return _read_xpt_bytes(fh.read())
    for url in _nhanes_urls(file, cycle):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UroSentinel research"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            if not raw.startswith(b"HEADER RECORD"):   # an HTML error page, not a data file
                continue
            with open(path, "wb") as fh:
                fh.write(raw)
            return _read_xpt_bytes(raw)
        except Exception:
            continue
    return None


def _clean_codes(series: pd.Series) -> pd.Series:
    return series.where(~series.isin(NHANES_NONRESPONSE))


def harmonise_nhanes(merged: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """Map raw NHANES variable names to UroSentinel's harmonised columns.

    Variable names change slightly between cycles (for example night-time urination is KIQ480
    in older cycles and KIQ481 in 2021-2023), so each field tries the known names in turn.
    """
    def pick(*names):
        for n in names:
            if n in merged.columns:
                return merged[n]
        return pd.Series(np.nan, index=merged.index)

    out = pd.DataFrame(index=merged.index)
    out["seqn"] = merged["SEQN"].astype("Int64")
    out["cycle"] = cycle
    out["age"] = pick("RIDAGEYR")
    out["sex"] = pick("RIAGENDR").map({1: "Male", 2: "Female"})
    out["ethnicity"] = pick("RIDRETH1").map(ETHNICITY_LABELS)
    out["income_ratio"] = pick("INDFMPIR")
    out["survey_weight"] = pick("WTINT2YR")
    out["pregnant"] = pick("RIDEXPRG").map({1: 1, 2: 0, 3: np.nan})
    out["bmi"] = pick("BMXBMI")
    out["diabetes"] = _clean_codes(pick("DIQ010")).map({1: 1, 2: 0, 3: 0.5})   # 0.5 = borderline
    out["kidney_weak"] = _clean_codes(pick("KIQ022")).map({1: 1, 2: 0})
    out["leak_freq"] = _clean_codes(pick("KIQ005")) - 1             # 0 never ... 4 every day/night
    out["stress_leak"] = _clean_codes(pick("KIQ042")).map({1: 1, 2: 0})
    out["urge_leak"] = _clean_codes(pick("KIQ044")).map({1: 1, 2: 0})
    out["nocturia"] = _clean_codes(pick("KIQ481", "KIQ480"))         # times per night, 5 = 5 or more
    out.attrs["provenance"] = [f"NHANES {cycle} (CDC, public): DEMO, KIQ_U, DIQ, BMX"]
    out.attrs["synthetic"] = False
    return out


def load_nhanes(cycles: Iterable[str] = ("2015-2016", "2017-2018", "2021-2023"),
                cache_dir: str = "nhanes_cache", allow_synthetic_fallback: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """Download, merge and harmonise several NHANES cycles into one adult table.

    If the CDC website cannot be reached (for example in a locked-down network), a clearly
    labelled NHANES-shaped synthetic sample is returned instead, so the rest of the notebook can
    still be explored. Its ``attrs['synthetic']`` is True and every report says so.
    """
    frames, provenance = [], []
    for cycle in cycles:
        parts = {f: download_nhanes_file(f, cycle, cache_dir) for f in NHANES_FILES}
        if parts["DEMO"] is None or parts["KIQ_U"] is None:
            if verbose:
                print(f"  NHANES {cycle}: could not download, skipped")
            continue
        merged = parts["DEMO"]
        for f in ("KIQ_U", "DIQ", "BMX"):
            if parts[f] is not None:
                merged = merged.merge(parts[f], on="SEQN", how="left")
        h = harmonise_nhanes(merged, cycle)
        provenance += h.attrs["provenance"]
        frames.append(h)
        if verbose:
            print(f"  NHANES {cycle}: {len(h):,} participants")
    if not frames:
        if not allow_synthetic_fallback:
            raise RuntimeError("No NHANES cycles could be downloaded.")
        if verbose:
            print("  NHANES could not be reached. Using a SYNTHETIC NHANES-shaped sample for demonstration.")
        return synthetic_nhanes_like()
    df = pd.concat(frames, ignore_index=True)
    # Survey weights: when cycles are pooled, the CDC advises dividing each 2-year weight by the
    # number of cycles. The 2021-2023 cycle covers about 2 years too, so the same rule is used.
    df["survey_weight"] = df["survey_weight"] / len(frames)
    df.attrs["provenance"] = provenance
    df.attrs["synthetic"] = False
    return df


def synthetic_nhanes_like(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """A synthetic sample with the same columns as harmonised NHANES. For offline demos only.

    Relationships are set to be broadly plausible (urge leakage rises with age, female sex, BMI,
    diabetes and night-time urination), but the numbers are invented and must not be reported.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 81, n)
    sex = rng.choice(["Male", "Female"], n)
    bmi = np.clip(rng.normal(29, 6.5, n), 16, 60)
    diabetes = rng.choice([0, 0.5, 1], n, p=[0.82, 0.04, 0.14])
    kidney = (rng.random(n) < 0.04).astype(float)
    female = (sex == "Female").astype(float)
    nocturia = np.clip(np.round(rng.poisson(0.4 + 0.025 * (age - 20) + 0.6 * diabetes)), 0, 5)
    stress = (rng.random(n) < 1 / (1 + np.exp(-(-2.3 + 1.6 * female + 0.03 * (bmi - 29) + 0.015 * (age - 50))))).astype(float)
    logit = -3.4 + 0.035 * (age - 20) + 0.7 * female + 0.04 * (bmi - 29) + 0.6 * diabetes + 0.45 * nocturia + 1.1 * stress + 0.5 * kidney
    urge = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    leak_freq = np.where((urge + stress) > 0, rng.choice([1, 2, 3, 4], n, p=[0.35, 0.3, 0.2, 0.15]), 0).astype(float)
    df = pd.DataFrame({
        "seqn": np.arange(1, n + 1), "cycle": "synthetic", "age": age, "sex": sex,
        "ethnicity": rng.choice(list(ETHNICITY_LABELS.values()), n, p=[0.15, 0.1, 0.4, 0.22, 0.13]),
        "income_ratio": np.clip(rng.gamma(2.2, 1.2, n), 0, 5), "survey_weight": 1.0, "pregnant": 0.0,
        "bmi": bmi, "diabetes": diabetes, "kidney_weak": kidney, "leak_freq": leak_freq,
        "stress_leak": stress, "urge_leak": urge, "nocturia": nocturia,
    })
    # Add a little realistic messiness so the cleaning step has something to find.
    miss = rng.random(n) < 0.03
    df.loc[miss, "bmi"] = np.nan
    df.loc[rng.random(n) < 0.01, "diabetes"] = np.nan
    df.attrs["provenance"] = ["SYNTHETIC NHANES-shaped sample (generated offline, not real data)"]
    df.attrs["synthetic"] = True
    return df


# Target and features for the population screening task. The target uses urge leakage and how
# often leaks happen, so BOTH are kept out of the inputs. This removes the "partial feature
# overlap" limitation reported in the ABH 2026 OAB paper.
NHANES_TARGET = "oab_wet_proxy"
NHANES_FEATURES = ["age", "female", "bmi", "diabetes", "kidney_weak", "stress_leak", "nocturia"]
NHANES_AUDIT_ONLY = ["ethnicity", "income_ratio"]   # used to check fairness, never as inputs


def add_nhanes_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the OAB-wet proxy label and model-ready columns.

    NHANES has no direct urgency question. Following the ICS idea of urgency with leakage, the
    label is: leaked before reaching the toilet (KIQ044) AND leaks at least a few times a month
    (KIQ005). It is a symptom-based screening label, not a diagnosis.
    """
    out = df.copy()
    out["female"] = (out["sex"] == "Female").astype(float).where(out["sex"].notna())
    label = (out["urge_leak"] == 1) & (out["leak_freq"] >= 2)
    out[NHANES_TARGET] = label.astype(float).where(out["urge_leak"].notna() & out["leak_freq"].notna())
    return out


# ---------------------------------------------------------------------------------------------
# International: UCI Acute Inflammations
# ---------------------------------------------------------------------------------------------

UCI_COLUMNS = ["temperature", "nausea", "lumbar_pain", "urine_pushing", "micturition_pain",
               "urethral_burning", "bladder_inflammation", "nephritis"]


def load_uci_acute_inflammations(verbose: bool = True) -> pd.DataFrame | None:
    """Load the UCI Acute Inflammations dataset (120 patients, Poland). Returns None if offline."""
    df = None
    try:
        from ucimlrepo import fetch_ucirepo   # pip install ucimlrepo
        ds = fetch_ucirepo(id=184)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = UCI_COLUMNS
    except Exception:
        try:
            url = "https://archive.ics.uci.edu/static/public/184/acute+inflammations.zip"
            with urllib.request.urlopen(url, timeout=60) as resp:
                zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            raw = zf.read("diagnosis.data").decode("utf-16")
            rows = [r.split("\t") for r in raw.strip().splitlines()]
            df = pd.DataFrame(rows, columns=UCI_COLUMNS)
            df["temperature"] = df["temperature"].str.replace(",", ".").astype(float)
        except Exception:
            if verbose:
                print("  UCI Acute Inflammations could not be downloaded (offline?). Skipped.")
            return None
    for c in UCI_COLUMNS[1:]:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(float)
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df.attrs["provenance"] = ["UCI Machine Learning Repository, Acute Inflammations (id 184), Czerniak and Zarzycki"]
    df.attrs["synthetic"] = False
    if verbose:
        print(f"  UCI Acute Inflammations: {len(df)} patients")
    return df


# ---------------------------------------------------------------------------------------------
# UK harmonised data (user supplied, approved access only)
# ---------------------------------------------------------------------------------------------

UK_REQUIRED = ["age", "sex", "bmi", "diabetes", "kidney_weak", "stress_leak", "urge_leak",
               "leak_freq", "nocturia"]

UK_GUIDANCE = """UK data needs approved access, so UroSentinel never downloads it.
  - UK Biobank: apply at ukbiobank.ac.uk (approved researchers, analysis in its Research Analysis Platform)
  - CPRD (primary care records): apply through the CPRD research data governance process
  - OpenSAFELY: code runs inside the platform; only aggregate results leave
  - Local NHS audit: needs Caldicott Guardian approval and a DPIA
Export an anonymised CSV with these columns, then call load_uk_harmonised(path):
  """ + ", ".join(UK_REQUIRED) + """
  Optional: ethnicity, income_ratio or IMD_decile (deprivation), survey_weight, source
Coding: sex Male/Female; yes/no as 1/0; diabetes 1/0.5/0; leak_freq 0 (never) to 4 (daily);
nocturia 0 to 5 (times per night, 5 = 5 or more)."""


def load_uk_harmonised(path: str, source_name: str = "UK dataset") -> pd.DataFrame:
    """Load a UK CSV already mapped to the harmonised columns, and check it."""
    df = pd.read_csv(path)
    missing = [c for c in UK_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\n\n{UK_GUIDANCE}")
    if "IMD_decile" in df.columns and "income_ratio" not in df.columns:
        df["income_ratio"] = df["IMD_decile"]          # deprivation, used for the fairness audit only
    df["cycle"] = df.get("source", source_name)
    df.attrs["provenance"] = [f"{source_name} (user supplied, approved access): {os.path.basename(path)}"]
    df.attrs["synthetic"] = False
    return df


# ---------------------------------------------------------------------------------------------
# Clinical differential cohort (six conditions, Model 2 layout)
# ---------------------------------------------------------------------------------------------

CONDITIONS = ["OAB", "IC_BPS", "Acute_Cystitis", "Chronic_Cystitis", "Neurogenic_Bladder", "BOO_BPH"]

CLINICAL_FEATURES = ["Age", "Female", "Frequency_Score", "Urgency_Score", "Nocturia_Score",
                     "Pain_Score", "QoL_Impact_Score", "Duration_Months", "Fever", "Dysuria",
                     "Haematuria", "Nitrite_Positive", "Neuro_History", "PVR_ml",
                     "Recurrent_UTI_Count", "Weak_Stream"]

# Typical profile of each condition: mean symptom scores (0-10) and the chance of yes/no signs.
# Values are illustrative and broadly follow the clinical picture in the ABH 2026 RST paper.
_PROFILES = {
    #                     freq urg noct pain  dur(mo) fever dysur haem nitr neuro pvr  rUTI weak  female age
    "OAB":               (7.5, 7.9, 6.6, 1.8,  30,    .01,  .05,  .03, .03, .04,  40,  0.3, .10,  .70,  58),
    "IC_BPS":            (7.8, 5.4, 4.6, 7.9,  48,    .01,  .35,  .10, .03, .03,  35,  0.8, .05,  .85,  45),
    "Acute_Cystitis":    (7.0, 6.5, 3.5, 5.0,  0.3,   .20,  .90,  .30, .70, .02,  30,  0.5, .03,  .85,  38),
    "Chronic_Cystitis":  (6.0, 5.5, 3.5, 4.0,  14,    .05,  .60,  .15, .40, .05,  60,  3.5, .10,  .80,  55),
    "Neurogenic_Bladder":(6.0, 6.5, 5.5, 2.0,  40,    .05,  .15,  .05, .20, .90, 220,  2.0, .50,  .45,  50),
    "BOO_BPH":           (6.5, 5.0, 6.0, 1.5,  36,    .01,  .10,  .08, .05, .03, 160,  0.5, .85,  .00,  68),
}


def generate_clinical_cohort(n: int = 1800, seed: int = 42, add_messiness: bool = True) -> pd.DataFrame:
    """Create a SYNTHETIC six-condition cohort in the Model 2 layout, for demonstration only.

    It lets the full pipeline (cleaning, rough sets, models, SHAP, LIME, app) run before real
    clinical data is approved. Results on it show that the method works, not how well it would
    work on patients. With ``add_messiness`` it also adds duplicates, out-of-range scores,
    missing values and a few contradictory records, so the ethics-by-design cleaning and the
    rough set inconsistency check have something real to find.
    """
    rng = np.random.default_rng(seed)
    per = n // len(CONDITIONS)
    rows = []
    for cond in CONDITIONS:
        f, u, no, p, dur, fev, dys, hae, nit, neu, pvr, rut, weak, fem, age = _PROFILES[cond]
        k = per
        sc = lambda m: np.clip(np.round(rng.normal(m, 1.4, k)), 0, 10)
        female = (rng.random(k) < fem).astype(int)
        rows.append(pd.DataFrame({
            "Age": np.clip(np.round(rng.normal(age, 12, k)), 18, 95),
            "Female": female,
            "Frequency_Score": sc(f), "Urgency_Score": sc(u), "Nocturia_Score": sc(no),
            "Pain_Score": sc(p),
            "QoL_Impact_Score": np.clip(np.round((sc(f) + sc(p) + sc(u)) / 3 + rng.normal(0, 1, k)), 0, 10),
            "Duration_Months": np.round(np.clip(rng.lognormal(np.log(max(dur, 0.2)), 0.6, k), 0.1, 240), 1),
            "Fever": (rng.random(k) < fev).astype(int),
            "Dysuria": (rng.random(k) < dys).astype(int),
            "Haematuria": (rng.random(k) < hae).astype(int),
            "Nitrite_Positive": (rng.random(k) < nit).astype(int),
            "Neuro_History": (rng.random(k) < neu).astype(int),
            "PVR_ml": np.round(np.clip(rng.normal(pvr, pvr * 0.35, k), 0, 900)),
            "Recurrent_UTI_Count": rng.poisson(rut, k),
            "Weak_Stream": (rng.random(k) < weak).astype(int),
            "Primary_Diagnosis": cond,
        }))
    df = pd.concat(rows, ignore_index=True)
    # Benign prostatic obstruction only occurs in men in this cohort.
    df.loc[df["Primary_Diagnosis"] == "BOO_BPH", "Female"] = 0
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "Patient_ID", [f"SYN{i:05d}" for i in range(len(df))])
    if add_messiness:
        dup = df.sample(12, random_state=1)
        bad = df.sample(6, random_state=2).index
        df.loc[bad, "Pain_Score"] = 14          # outside the 0-10 scale
        df.loc[df.sample(25, random_state=3).index, "PVR_ml"] = np.nan
        df.loc[df.sample(3, random_state=4).index, "Age"] = 160   # impossible age
        contra = df.sample(8, random_state=5).copy()
        contra["Primary_Diagnosis"] = "Acute_Cystitis"            # same symptoms, different label
        contra["Patient_ID"] = contra["Patient_ID"] + "-B"
        df = pd.concat([df, dup, contra], ignore_index=True)
    df.attrs["provenance"] = ["SYNTHETIC six-condition clinical cohort (illustrative profiles, not real patients)"]
    df.attrs["synthetic"] = True
    return df


def load_model2_csv(path: str) -> pd.DataFrame:
    """Load a CSV in the original Model 2 layout and add any newer columns as missing.

    Model 2 columns: Patient_ID, Age, Sex, Primary_Diagnosis, Frequency_Score, Urgency_Score,
    Nocturia_Score, Pain_Score, QoL_Impact_Score. Newer optional columns are listed in
    CLINICAL_FEATURES; any that are absent are filled in during cleaning or dropped.
    """
    df = pd.read_csv(path)
    if "Female" not in df.columns and "Sex" in df.columns:
        df["Female"] = df["Sex"].astype(str).str.strip().str.lower().map(
            {"female": 1, "f": 1, "male": 0, "m": 0})
    for c in CLINICAL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df.attrs["provenance"] = [f"User clinical CSV (Model 2 layout): {os.path.basename(path)}"]
    df.attrs["synthetic"] = False
    return df
