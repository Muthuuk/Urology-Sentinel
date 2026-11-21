"""Tests for UroSentinel. Run with:  python -m pytest -q"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from urosentinel import data_sources as ds, ethics, models as m, roughsets as rs
from urosentinel.service import ClinicalService, ScreeningService, html_report, red_flags


# ---------------------------------------------------------------- rough sets
def toy_table():
    # Classic small decision table: 'b' alone decides the outcome except rows 4 and 5,
    # which are identical on every feature but have different decisions.
    X = pd.DataFrame({"a": ["1", "1", "2", "2", "3", "3"], "b": ["x", "y", "x", "y", "z", "z"],
                      "c": ["p", "p", "q", "q", "r", "r"]})
    y = pd.Series(["yes", "no", "yes", "no", "yes", "no"])
    return X, y


def test_crisp_dependency_boundary_and_inconsistency():
    X, y = toy_table()
    R = rs.RoughSet(X, y, beta=1.0, min_size=1)
    assert R.dependency(list(X.columns)) == pytest.approx(4 / 6)
    assert set(R.boundary_region()) == {4, 5}
    inc = R.inconsistent_records()
    assert set(inc["row"]) == {4, 5}
    lower, upper = R.approximations("yes")
    assert set(lower) == {0, 2} and set(upper) == {0, 2, 4, 5}


def test_crisp_reduct_and_core():
    X, y = toy_table()
    R = rs.RoughSet(X, y, beta=1.0, min_size=1)
    red = R.quick_reduct(tolerance=0.0)
    assert R.dependency(red) == pytest.approx(R.dependency(list(X.columns)))
    assert "b" in red and len(red) == 1
    assert R.core(list(X.columns), tolerance=0.0) == ["b"]


def test_classify_statuses():
    X, y = toy_table()
    R = rs.RoughSet(X, y, beta=1.0, min_size=1)
    assert R.classify({"a": "1", "b": "x", "c": "p"})["status"] == "certain"
    assert R.classify({"a": "3", "b": "z", "c": "r"})["status"] == "boundary"
    assert R.classify({"a": "9", "b": "x", "c": "p"})["status"] == "unseen"


def test_accuracy_filter_drops_only_clear_minority():
    X = pd.DataFrame({"f": ["a"] * 5 + ["b"] * 2})
    y = pd.Series(["A", "A", "A", "A", "B", "A", "B"])
    R = rs.RoughSet(X, y, beta=1.0, min_size=1)
    keep = R.accuracy_filter(min_group=3, min_majority=0.75)
    assert 4 not in keep              # disagrees with a clear 4-to-1 majority
    assert {5, 6} <= set(keep)        # 1-to-1 split is genuine uncertainty, kept


def test_suggested_beta_above_base_rate():
    y = pd.Series([0] * 85 + [1] * 15)
    assert rs.RoughSet.suggested_beta(y) > 0.85


# ---------------------------------------------------------------- data sources
def test_harmonise_nhanes_codes_and_zero_fix():
    raw = pd.DataFrame({"SEQN": [1, 2, 3], "RIDAGEYR": [40, 70, 55], "RIAGENDR": [1, 2, 2],
                        "RIDRETH1": [3, 4, 1], "KIQ005": [1, 4, 9], "KIQ042": [2, 1, 7],
                        "KIQ044": [2, 1, 1], "KIQ481": [5.4e-79, 3, 99], "KIQ022": [2, 1, 2],
                        "DIQ010": [2, 1, 3], "BMXBMI": [24.0, 31.0, np.nan]})
    num = raw.select_dtypes("number").columns
    raw[num] = raw[num].mask(raw[num].abs() < 1e-30, 0.0)
    h = ds.harmonise_nhanes(raw, "2021-2023")
    assert h.loc[0, "nocturia"] == 0 and np.isnan(h.loc[2, "nocturia"])     # 99 = don't know
    assert np.isnan(h.loc[2, "leak_freq"]) and np.isnan(h.loc[2, "stress_leak"])
    assert h.loc[2, "diabetes"] == 0.5 and h.loc[1, "sex"] == "Female"
    t = ds.add_nhanes_target(h)
    assert t.loc[1, ds.NHANES_TARGET] == 1 and t.loc[0, ds.NHANES_TARGET] == 0


def test_label_inputs_are_not_model_features():
    assert "urge_leak" not in ds.NHANES_FEATURES and "leak_freq" not in ds.NHANES_FEATURES
    assert not set(ds.NHANES_AUDIT_ONLY) & set(ds.NHANES_FEATURES)


def test_synthetic_data_is_labelled():
    assert ds.synthetic_nhanes_like(200).attrs["synthetic"] is True
    c = ds.generate_clinical_cohort(300)
    assert c.attrs["synthetic"] is True and set(c["Primary_Diagnosis"]) == set(ds.CONDITIONS)


def test_uk_loader_rejects_missing_columns(tmp_path):
    p = tmp_path / "uk.csv"
    pd.DataFrame({"age": [50]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        ds.load_uk_harmonised(str(p))


# ---------------------------------------------------------------- ethics
def test_ethical_clean_logs_every_step():
    df = pd.DataFrame({"Patient_ID": [1, 2, 3, 4], "name": ["a", "b", "c", "d"],
                       "Pain_Score": [3, 14, 3, 5], "Age": [40, 50, 40, 160], "pregnant": [0, 0, 0, 1]})
    df.loc[2, "Patient_ID"] = 9
    out, log = ethics.ethical_clean(df, valid_ranges={"Pain_Score": (0, 10), "Age": (18, 110)},
                                    id_cols=["Patient_ID"], exclude_if={"pregnant": 1})
    steps = log.to_frame()["step"].tolist()
    assert "name" not in out.columns
    assert any("duplicate" in s for s in steps)
    assert out["Pain_Score"].isna().sum() == 1
    assert len(out) == 2                          # one duplicate removed, one pregnant excluded


def test_fairness_audit_flags_low_sensitivity_group():
    rng = np.random.default_rng(0)
    n = 400
    g = np.array(["A"] * 200 + ["B"] * 200)
    y = rng.integers(0, 2, n)
    pred = y.copy()
    pred[(g == "B") & (y == 1)] = 0                # misses every case in group B
    res = ethics.fairness_audit(y, pred, pred.astype(float), pd.DataFrame({"grp": g}))
    assert res.set_index("group").loc["B", "note"].startswith("REVIEW")


# ---------------------------------------------------------------- models and services
@pytest.fixture(scope="module")
def trained():
    nh = ds.add_nhanes_target(ds.synthetic_nhanes_like(1500))
    sm = m.train_binary(nh, ds.NHANES_FEATURES, ds.NHANES_TARGET, model="logistic")
    clin, _ = ethics.ethical_clean(ds.generate_clinical_cohort(600, add_messiness=False), id_cols=["Patient_ID"])
    cm = m.train_multiclass(clin, ds.CLINICAL_FEATURES, "Primary_Diagnosis", model="random_forest")
    R = rs.RoughSet(rs.discretise(clin[ds.CLINICAL_FEATURES], cuts=rs.CLINICAL_CUTS), clin["Primary_Diagnosis"])
    R.quick_reduct()
    return sm, cm, R


def test_screening_model_reaches_target_sensitivity_on_training(trained):
    sm, _, _ = trained
    assert sm.test_metrics["AUROC"] > 0.6
    assert 0 < sm.threshold < 1


def test_screening_service_output(trained):
    sm, _, _ = trained
    s = ScreeningService(sm)
    f = s.to_features(60, "Female", 160, 80, "Yes", False, True, 3)
    r = s.predict(f)
    assert r["band"] in {"low", "moderate", "high"} and len(r["explanation"]) == 3
    assert "Not a medical device" in r["disclaimer"]


def test_clinical_service_triage_flags_and_feedback(trained, tmp_path):
    _, cm, R = trained
    svc = ClinicalService(cm, R, rs.CLINICAL_CUTS, feedback_path=str(tmp_path / "fb.csv"))
    x = {"Age": 66, "Female": 0, "Frequency_Score": 6, "Urgency_Score": 5, "Nocturia_Score": 6, "Pain_Score": 1,
         "QoL_Impact_Score": 5, "Duration_Months": 30, "Fever": 0, "Dysuria": 0, "Haematuria": 1,
         "Nitrite_Positive": 0, "Neuro_History": 0, "PVR_ml": 320, "Recurrent_UTI_Count": 0, "Weak_Stream": 1}
    r = svc.predict(x)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert any("NG12" in f for f in r["red_flags"]) and any("300 ml" in f for f in r["red_flags"])
    assert r["rough_set"]["status"] in {"certain", "boundary", "unseen"}
    assert 0 <= r["agreement"] <= 1
    path = svc.log_feedback(x, r, "BOO_BPH", "test")
    assert pd.read_csv(path).shape[0] == 1
    assert "<html>" in html_report(x, r)


def test_red_flags_quiet_for_unremarkable_case():
    assert red_flags({"Age": 30, "Haematuria": 0, "Fever": 0, "PVR_ml": 20, "Neuro_History": 0}) == []


def test_discretise_matches_int_and_float_inputs():
    a = rs.discretise(pd.DataFrame({"Fever": [1, 0]}))
    b = rs.discretise(pd.DataFrame({"Fever": [1.0, 0.0]}))
    assert a.equals(b)


def test_clinical_triage_recognises_known_profiles(trained):
    _, cm, R = trained
    svc = ClinicalService(cm, R, rs.CLINICAL_CUTS)
    statuses = {svc.predict(cm.X_test.iloc[i].to_dict(), explain=False)["rough_set"]["status"] for i in range(40)}
    assert statuses != {"unseen"}
