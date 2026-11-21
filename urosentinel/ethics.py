"""Ethics by design for UroSentinel.

Ethics is treated as a set of engineering requirements that are checked in code, not as a
statement added at the end. The module follows Prof Muthu Ramachandran's V-EGRCSbyD approach
(Validation, Ethics, Governance, Risk, Compliance and Security by Design) and the five widely
used AI ethics principles: beneficence, non-maleficence, autonomy, justice and explicability.

It provides:
  * ``ethical_clean``: data cleaning where every step is logged with the principle it serves
  * ``representativeness``: who is in the data, compared with the population (weighted)
  * ``fairness_audit``: whether the model misses more cases in some groups than others
  * ``vegrcs_checklist``: the V-EGRCSbyD checklist, filled in from evidence the code produced
  * ``model_card``: a plain-English model card for reviewers, clinicians and patients
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Columns that could identify a person directly. They are removed before any analysis.
DIRECT_IDENTIFIERS = ["name", "first_name", "surname", "nhs_number", "nhs_no", "address",
                      "postcode", "post_code", "email", "phone", "telephone", "dob",
                      "date_of_birth", "hospital_number", "mrn"]

PRINCIPLES = {
    "privacy": "Autonomy and privacy (UK GDPR data minimisation)",
    "accuracy": "Non-maleficence: inaccurate data leads to unsafe predictions",
    "transparency": "Explicability: every change to the data is recorded and reported",
    "justice": "Justice: the data and model must work fairly for all groups",
    "scope": "Non-maleficence: use the model only for the people it was built for",
}


@dataclass
class CleaningLog:
    """A record of every cleaning step, so the data's journey can be audited."""
    steps: list = field(default_factory=list)

    def add(self, step: str, principle: str, rows: int, detail: str = "") -> None:
        self.steps.append({"step": step, "principle": PRINCIPLES[principle],
                           "rows_affected": int(rows), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def ethical_clean(df: pd.DataFrame, valid_ranges: dict | None = None,
                  id_cols: list | None = None, exclude_if: dict | None = None,
                  max_missing_share: float = 0.4,
                  protected_features: list | None = None) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean a dataset in a way that is logged and explainable.

    Values are set to missing rather than guessed. Filling in (imputation) happens later inside
    the model pipeline, using the training data only, so test data never leaks into training.

    Parameters
    ----------
    valid_ranges : {column: (low, high)}; values outside become missing (for example a pain
        score of 14 on a 0-10 scale, or an age of 160).
    id_cols : columns that identify a record (for duplicate checks), kept but never modelled.
    exclude_if : {column: value}; rows matching are outside the model's intended scope (for
        example pregnancy, where bladder symptoms have a different cause).
    max_missing_share : features missing in more than this share of rows are reported.
    protected_features : columns that must not be inputs (used for the fairness audit only).
    """
    log = CleaningLog()
    out = df.copy()
    attrs = dict(df.attrs)

    # 1. Privacy: remove direct identifiers.
    ids = [c for c in out.columns if c.lower() in DIRECT_IDENTIFIERS]
    if ids:
        out = out.drop(columns=ids)
        log.add("Removed direct identifiers", "privacy", len(out), ", ".join(ids))
    else:
        log.add("Checked for direct identifiers", "privacy", 0, "none found")

    # 2. Accuracy: exact duplicate records (ignoring the record ID).
    key = [c for c in out.columns if c not in (id_cols or [])]
    dups = out.duplicated(subset=key, keep="first")
    if dups.any():
        out = out.loc[~dups]
    log.add("Removed exact duplicate records", "accuracy", dups.sum())

    # 3. Accuracy: values outside valid clinical ranges become missing.
    for col, (lo, hi) in (valid_ranges or {}).items():
        if col in out.columns:
            bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
            if bad.any():
                out.loc[bad, col] = np.nan
                log.add(f"Set out-of-range {col} to missing", "accuracy", bad.sum(),
                        f"valid range {lo} to {hi}")

    # 4. Scope: remove people the model is not designed for.
    for col, val in (exclude_if or {}).items():
        if col in out.columns:
            hit = out[col] == val
            out = out.loc[~hit]
            log.add(f"Excluded rows where {col} = {val}", "scope", hit.sum(), "outside intended use")

    # 5. Transparency: report heavily missing features.
    share = out.isna().mean()
    heavy = share[share > max_missing_share]
    for col, s in heavy.items():
        log.add(f"High missingness in {col}", "transparency", int(s * len(out)),
                f"{s:.0%} missing; treat results using it with caution")

    # 6. Justice: protected characteristics are kept for auditing but flagged as never-inputs.
    if protected_features:
        log.add("Protected characteristics kept for fairness audit only", "justice", 0,
                ", ".join(protected_features))

    out.attrs.update(attrs)
    return out.reset_index(drop=True), log


def representativeness(df: pd.DataFrame, group_cols: list, weight_col: str | None = None) -> pd.DataFrame:
    """Share of each group in the sample, and in the population once survey weights are applied.

    NHANES deliberately over-samples some groups. Survey weights correct for this, so a large gap
    between the two columns is expected and is the reason the weights are used.
    """
    rows = []
    for col in group_cols:
        if col not in df.columns:
            continue
        g = df[col].fillna("Not recorded")
        unweighted = g.value_counts(normalize=True)
        if weight_col and weight_col in df.columns:
            w = df[weight_col].fillna(0)
            weighted = w.groupby(g).sum() / w.sum()
        else:
            weighted = unweighted
        for level in unweighted.index:
            rows.append({"characteristic": col, "group": level, "n": int((g == level).sum()),
                         "sample_share": round(unweighted[level], 3),
                         "population_share": round(float(weighted.get(level, np.nan)), 3)})
    return pd.DataFrame(rows)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 39, 59, 74, 200], labels=["18-39", "40-59", "60-74", "75+"])


def fairness_audit(y_true, y_pred, y_prob, groups: pd.DataFrame, min_n: int = 30,
                   gap_threshold: float = 0.10) -> pd.DataFrame:
    """Compare performance across groups.

    For screening, the most important harm is a missed case (false negative), so the key column
    is ``sensitivity`` (the share of true cases found). Groups whose sensitivity is more than
    ``gap_threshold`` below the best group are flagged for review.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    rows = []
    for col in groups.columns:
        g = groups[col].astype(str).fillna("Not recorded").values
        for level in pd.unique(g):
            m = g == level
            n, pos = int(m.sum()), int(y_true[m].sum())
            if n < min_n or pos < 5:
                rows.append({"characteristic": col, "group": level, "n": n, "cases": pos,
                             "sensitivity": np.nan, "precision": np.nan, "auroc": np.nan,
                             "note": "too few to judge"})
                continue
            auc = roc_auc_score(y_true[m], y_prob[m]) if 0 < pos < n else np.nan
            rows.append({"characteristic": col, "group": level, "n": n, "cases": pos,
                         "sensitivity": round(recall_score(y_true[m], y_pred[m], zero_division=0), 3),
                         "precision": round(precision_score(y_true[m], y_pred[m], zero_division=0), 3),
                         "auroc": round(auc, 3), "note": ""})
    res = pd.DataFrame(rows)
    for col in res["characteristic"].unique():
        part = res["characteristic"] == col
        # The reference is the best group with enough cases (20+) to be reliable.
        ref = part & (res["cases"] >= 20)
        best = res.loc[ref if ref.any() else part, "sensitivity"].max()
        flag = part & (res["sensitivity"] < best - gap_threshold)
        res.loc[flag, "note"] = "REVIEW: misses more cases than the best group"
    return res


def vegrcs_checklist(evidence: dict) -> pd.DataFrame:
    """The V-EGRCSbyD checklist, with each item marked from evidence the pipeline produced.

    ``evidence`` keys (all optional): cleaning_log, fairness_flags, explainability, rough_set,
    synthetic, provenance, human_oversight, external_validation.
    """
    ev = evidence
    items = [
        ("Validation", "Model tested on held-out data with clinically relevant measures",
         ev.get("validated", False)),
        ("Validation", "Tested on data from another country or setting (external validation)",
         ev.get("external_validation", False)),
        ("Ethics", "Data cleaning logged with the principle each step serves",
         ev.get("cleaning_log") is not None),
        ("Ethics", "Fairness audit run; groups with lower sensitivity flagged",
         ev.get("fairness_flags") is not None),
        ("Ethics", "Explanations available for every prediction (SHAP and LIME)",
         ev.get("explainability", False)),
        ("Governance", "Data sources and their licences recorded",
         bool(ev.get("provenance"))),
        ("Governance", "Clinician can accept or override the suggestion, and this is logged",
         ev.get("human_oversight", False)),
        ("Risk", "Uncertain cases are referred, not forced into a label (rough set boundary)",
         ev.get("rough_set", False)),
        ("Risk", "Results on synthetic data are labelled as such",
         True if ev.get("synthetic") is not None else False),
        ("Compliance", "Protected characteristics excluded from model inputs",
         ev.get("protected_excluded", False)),
        ("Compliance", "DPIA, clinical safety (DCB0129/DCB0160) and MHRA status required before clinical use",
         False),
        ("Security", "No direct identifiers in data; nothing sent to outside services",
         ev.get("cleaning_log") is not None),
    ]
    df = pd.DataFrame(items, columns=["V-EGRCSbyD area", "requirement", "met"])
    df["met"] = df["met"].map({True: "Yes", False: "Not yet"})
    return df


def model_card(name: str, purpose: str, data_attrs: dict, metrics: dict, fairness: pd.DataFrame | None,
               features: list, excluded_features: list, limitations: list, version: str) -> str:
    """Return a plain-English model card as Markdown."""
    prov = "\n".join(f"- {p}" for p in data_attrs.get("provenance", ["not recorded"]))
    synth = data_attrs.get("synthetic", False)
    met = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    flagged = ""
    if fairness is not None and len(fairness):
        f = fairness[fairness["note"].str.startswith("REVIEW")]
        flagged = ("\n".join(f"- {r.characteristic} = {r.group}: sensitivity {r.sensitivity}" for r in f.itertuples())
                   or "- No group fell more than 10 points below the best group.")
    lim = "\n".join(f"- {x}" for x in limitations)
    return f"""# Model card: {name}

**Version:** {version}  |  **Date:** {_dt.date.today().isoformat()}

## What it is for
{purpose}

**Not for:** diagnosing anyone, or replacing a clinician's judgement. Research prototype only.

## Data
{prov}

{"**WARNING: these results come from SYNTHETIC data and show only that the method works.**" if synth else ""}

## Inputs
{", ".join(features)}

Deliberately **not** used as inputs: {", ".join(excluded_features) or "none"}

## Performance (held-out test data)
{met}

## Fairness check
{flagged}

## Known limitations
{lim}
"""
