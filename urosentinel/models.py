"""Model training and evaluation for UroSentinel.

Four model families are compared on equal terms (same data split, same seed):
  * logistic regression  (simple, easy to check, a fair baseline)
  * random forest        (the model used in "Urology prediction model 2")
  * gradient boosting    (usually the strongest on tabular data)
  * neural network (MLP) (the family used in the ABH 2026 OAB deep learning paper)

Missing values are filled in inside each pipeline using the training data only, so nothing from
the test set leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             balanced_accuracy_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_model(name: str, n_classes: int = 2) -> Pipeline:
    """Build a pipeline: fill missing values, scale if needed, then the classifier."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name == "logistic":
        steps += [("scale", StandardScaler()),
                  ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    elif name == "random_forest":
        steps += [("model", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1))]
    elif name == "gradient_boosting":
        steps += [("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                           class_weight="balanced", random_state=SEED))]
    elif name == "neural_network":
        steps += [("scale", StandardScaler()),
                  ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                          max_iter=600, random_state=SEED))]
    else:
        raise ValueError(f"Unknown model {name}")
    return Pipeline(steps)


MODEL_NAMES = ["logistic", "random_forest", "gradient_boosting", "neural_network"]


def _fit(pipe: Pipeline, X, y, w=None):
    if w is not None and pipe.steps[-1][0] == "model" and not isinstance(pipe[-1], MLPClassifier):
        return pipe.fit(X, y, model__sample_weight=w)
    return pipe.fit(X, y)


def binary_metrics(y, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"AUROC": roc_auc_score(y, prob), "AUPRC": average_precision_score(y, prob),
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "precision": precision_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0), "Brier": brier_score_loss(y, prob),
            "missed_cases": int(fn), "false_alarms": int(fp)}


def threshold_for_sensitivity(y, prob, target: float = 0.85) -> float:
    """Highest threshold that still finds at least ``target`` of true cases (screening rule)."""
    order = np.sort(np.unique(prob))[::-1]
    for t in order:
        if recall_score(y, (prob >= t).astype(int), zero_division=0) >= target:
            return float(t)
    return float(order[-1])


def compare_binary(X: pd.DataFrame, y, weights=None, models=MODEL_NAMES, cv: int = 5,
                   target_sensitivity: float = 0.85) -> pd.DataFrame:
    """Cross-validated comparison of the model families on training data."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    rows = []
    y = np.asarray(y)
    for name in models:
        prob = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            pipe = make_model(name)
            _fit(pipe, X.iloc[tr], y[tr], None if weights is None else np.asarray(weights)[tr])
            prob[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        t = threshold_for_sensitivity(y, prob, target_sensitivity)
        m = binary_metrics(y, prob, t)
        rows.append({"model": name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()},
                     "threshold": round(t, 3)})
    return pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    features: list
    threshold: float = 0.5
    classes: list = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_prob: np.ndarray | None = None
    test_index: np.ndarray | None = None   # row positions of the test set in the training table

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])


def train_binary(df: pd.DataFrame, features: list, target: str, model: str = "gradient_boosting",
                 weight_col: str | None = None, target_sensitivity: float = 0.85,
                 test_size: float = 0.2) -> TrainedModel:
    """Train on 80%, choose the screening threshold by cross-validation on the training part,
    then report results once on the untouched 20% test set."""
    d = df.dropna(subset=[target]).reset_index(drop=True)
    X, y = d[features], d[target].astype(int).to_numpy()
    w = d[weight_col].to_numpy() if weight_col else None
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=test_size, stratify=y, random_state=SEED)
    w_tr = None if w is None else w[idx_tr]
    # Threshold from out-of-fold predictions on the training data only.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(idx_tr))
    for tr, va in skf.split(idx_tr, y[idx_tr]):
        p = make_model(model)
        _fit(p, X.iloc[idx_tr[tr]], y[idx_tr[tr]], None if w_tr is None else w_tr[tr])
        oof[va] = p.predict_proba(X.iloc[idx_tr[va]])[:, 1]
    thr = threshold_for_sensitivity(y[idx_tr], oof, target_sensitivity)
    pipe = _fit(make_model(model), X.iloc[idx_tr], y[idx_tr], w_tr)
    prob = pipe.predict_proba(X.iloc[idx_te])[:, 1]
    metrics = {k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in binary_metrics(y[idx_te], prob, thr).items()}
    return TrainedModel(model, pipe, features, thr, [0, 1], metrics,
                        X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True),
                        y[idx_tr], y[idx_te], prob, idx_te)


def train_multiclass(df: pd.DataFrame, features: list, target: str, model: str = "gradient_boosting",
                     test_size: float = 0.25, keep_rows: np.ndarray | None = None) -> TrainedModel:
    """Train a multi-class differential model and report macro-averaged results."""
    d = df.dropna(subset=[target]).reset_index(drop=True)
    X, y = d[features], d[target].astype(str).to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=test_size, stratify=y, random_state=SEED)
    if keep_rows is not None:               # e.g. rough set accuracy filter, applied to training rows only
        idx_tr = np.intersect1d(idx_tr, keep_rows)
    pipe = make_model(model).fit(X.iloc[idx_tr], y[idx_tr])
    prob = pipe.predict_proba(X.iloc[idx_te])
    pred = pipe.classes_[prob.argmax(1)]
    metrics = {"accuracy": round(accuracy_score(y[idx_te], pred), 3),
               "balanced_accuracy": round(balanced_accuracy_score(y[idx_te], pred), 3),
               "macro_F1": round(f1_score(y[idx_te], pred, average="macro"), 3),
               "macro_AUROC": round(roc_auc_score(y[idx_te], prob, multi_class="ovr", average="macro"), 3)}
    return TrainedModel(model, pipe, features, 0.5, list(pipe.classes_), metrics,
                        X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True),
                        y[idx_tr], y[idx_te], prob, idx_te)


def external_validation(tm: TrainedModel, df: pd.DataFrame, target: str) -> dict:
    """Check a trained screening model on data from another setting (for example UK data)."""
    d = df.dropna(subset=[target])
    prob = tm.predict_proba(d)[:, 1]
    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in binary_metrics(d[target].astype(int).to_numpy(), prob, tm.threshold).items()}
