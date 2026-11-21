"""Explainability for UroSentinel: SHAP and LIME, with plain-English summaries.

SHAP (SHapley Additive exPlanations) comes from game theory. It shares out the credit for a
prediction fairly among the inputs, and adds up exactly to the model's output.

LIME (Local Interpretable Model-agnostic Explanations) fits a small, simple model around one
patient's answers to see which inputs matter most for that one prediction.

The two work in different ways, so when they agree on the top reasons we can trust the
explanation more. ``agreement`` measures this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plain-English names for features, used in patient and clinician summaries.
FEATURE_TEXT = {
    "age": "age", "female": "being female", "bmi": "body weight (BMI)", "diabetes": "diabetes",
    "kidney_weak": "a kidney condition", "stress_leak": "leaking when coughing, sneezing or lifting",
    "nocturia": "getting up at night to pass urine",
    "Age": "age", "Female": "being female", "Frequency_Score": "how often urine is passed",
    "Urgency_Score": "sudden urgency", "Nocturia_Score": "night-time urination",
    "Pain_Score": "bladder pain", "QoL_Impact_Score": "impact on daily life",
    "Duration_Months": "how long symptoms have lasted", "Fever": "fever", "Dysuria": "pain or burning when passing urine",
    "Haematuria": "blood in the urine", "Nitrite_Positive": "a positive nitrite dipstick",
    "Neuro_History": "a neurological condition", "PVR_ml": "urine left after passing (post-void residual)",
    "Recurrent_UTI_Count": "number of past urine infections", "Weak_Stream": "a weak urine stream",
}


def _transform(tm, X: pd.DataFrame):
    """Return (values the final model sees, readable values after filling missing data)."""
    pre = tm.pipeline[:-1]
    readable = tm.pipeline.named_steps["impute"].transform(X[tm.features])
    return pre.transform(X[tm.features]), readable


def _is_tree(model) -> bool:
    return type(model).__name__ in {"RandomForestClassifier", "HistGradientBoostingClassifier",
                                    "GradientBoostingClassifier", "ExtraTreesClassifier"}


def shap_explain(tm, X: pd.DataFrame, class_index: int = 1, background_n: int = 200):
    """SHAP values for rows of ``X`` for one class. Returns a ``shap.Explanation``."""
    import shap
    final = tm.pipeline[-1]
    bg = tm.X_train.sample(min(background_n, len(tm.X_train)), random_state=0)
    bg_t, _ = _transform(tm, bg)
    X_t, X_read = _transform(tm, X)
    if _is_tree(final):
        explainer = shap.TreeExplainer(final, data=bg_t, model_output="probability") \
            if type(final).__name__ == "RandomForestClassifier" else shap.TreeExplainer(final)
        sv = explainer(X_t)
        values, base = sv.values, sv.base_values
    else:
        f = lambda Z: final.predict_proba(Z)
        explainer = shap.Explainer(f, shap.maskers.Independent(bg_t, max_samples=background_n),
                                   algorithm="permutation", seed=0)
        sv = explainer(X_t, max_evals=2 * X_t.shape[1] + 1 if X_t.shape[1] > 20 else "auto")
        values, base = sv.values, sv.base_values
    values, base = np.asarray(values), np.asarray(base)
    if values.ndim == 3:                       # (rows, features, classes)
        values, base = values[:, :, class_index], base[..., class_index] if base.ndim > 1 else base
    elif len(tm.classes) == 2 and class_index == 0:
        values, base = -values, -base
    base = np.broadcast_to(base, (values.shape[0],)).astype(float)
    return shap.Explanation(values=values, base_values=base, data=X_read, feature_names=tm.features)


def lime_explain(tm, row: pd.DataFrame, class_index: int = 1, num_features: int = 6):
    """LIME explanation of one patient. Returns a list of (condition, weight) pairs."""
    from lime.lime_tabular import LimeTabularExplainer
    _, train_read = _transform(tm, tm.X_train)
    _, row_read = _transform(tm, row)
    names = [str(c) for c in tm.classes]
    explainer = LimeTabularExplainer(train_read, feature_names=tm.features, class_names=names,
                                     mode="classification", discretize_continuous=True, random_state=0)
    fn = lambda A: tm.pipeline.predict_proba(pd.DataFrame(A, columns=tm.features))
    exp = explainer.explain_instance(row_read[0], fn, num_features=num_features, labels=[class_index])
    return exp.as_list(label=class_index), exp


def top_features_shap(explanation, row: int = 0, k: int = 3) -> list[tuple[str, float]]:
    v = explanation.values[row]
    order = np.argsort(-np.abs(v))[:k]
    return [(explanation.feature_names[i], float(v[i])) for i in order]


def _lime_feature(cond: str, features: list) -> str:
    """Pull the feature name out of a LIME condition such as '2.00 < nocturia <= 3.00'."""
    for f in sorted(features, key=len, reverse=True):
        if f in cond:
            return f
    return cond


def agreement(shap_top: list, lime_list: list, features: list, k: int = 3) -> float:
    """Share of SHAP's top-k features that also appear in LIME's top-k (0 to 1)."""
    s = {f for f, _ in shap_top[:k]}
    l = {_lime_feature(c, features) for c, _ in lime_list[:k]}
    return len(s & l) / max(len(s), 1)


def plain_english(top: list[tuple[str, float]], values: dict, audience: str = "patient") -> list[str]:
    """Turn the top SHAP contributions into short sentences."""
    lines = []
    for f, v in top:
        name = FEATURE_TEXT.get(f, f)
        direction = "raised" if v > 0 else "lowered"
        val = values.get(f)
        if audience == "patient":
            lines.append(f"Your answer about {name} {direction} the result.")
        else:
            lines.append(f"{name} = {val}: {direction} the score by {abs(v):.3f}")
    return lines
