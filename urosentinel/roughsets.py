"""Rough set theory (RST) for UroSentinel.

In plain terms: rough set theory groups together patients who look identical on the chosen
features (an "indiscernibility class"). If everyone in a group has the same diagnosis, the data
is certain about that group (the "lower approximation" or "positive region"). If a group holds
more than one diagnosis, the data cannot tell them apart; those patients sit in the
"boundary region".

UroSentinel uses this in four ways:
  1. Data accuracy: records that are identical on every feature but carry different labels are
     flagged as possibly wrong, so they can be checked or handled before training.
  2. Reduct: the smallest set of features that keeps the same power to separate the classes
     (feature selection that can be explained).
  3. Rules: transparent IF-THEN rules with certainty and support, readable by clinicians.
  4. Boundary triage: a new patient whose profile falls in a boundary group is referred for
     specialist review rather than given a falsely confident label.

Continuous values are first put into a few clinically meaningful bands (discretised).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCORE_BANDS = ([-0.1, 3, 6, 10.1], ["low", "mid", "high"])   # 0-10 symptom scores


def discretise(df: pd.DataFrame, cuts: dict | None = None, quantile_cols: list | None = None,
               q: int = 3) -> pd.DataFrame:
    """Turn numeric columns into bands.

    ``cuts`` gives clinical cut points as {column: (bin_edges, labels)}. ``quantile_cols`` are cut
    into ``q`` equal-sized groups. Other columns are kept as they are (already categorical).
    Missing values become the band "missing" so they still count as information.
    """
    out = pd.DataFrame(index=df.index)
    cuts = cuts or {}
    for col in df.columns:
        s = df[col]
        if col in cuts:
            edges, labels = cuts[col]
            out[col] = pd.cut(s, bins=edges, labels=labels).astype(object)
        elif quantile_cols and col in quantile_cols:
            out[col] = pd.qcut(s, q=q, labels=[f"q{i+1}" for i in range(q)], duplicates="drop").astype(object)
        else:
            out[col] = s.astype(object)
        out[col] = out[col].where(pd.notna(out[col]), "missing").astype(str)
    return out


@dataclass
class Rule:
    conditions: dict
    decision: str
    certainty: float
    support: int
    coverage: float

    def __str__(self) -> str:
        cond = " AND ".join(f"{k} = {v}" for k, v in self.conditions.items())
        return (f"IF {cond} THEN {self.decision}  "
                f"(certainty {self.certainty:.2f}, support {self.support}, coverage {self.coverage:.1%})")


class RoughSet:
    """A decision table: discretised features (conditions) plus one decision column.

    ``beta`` and ``min_size`` give the variable-precision form of rough sets (Ziarko). A group
    counts as certain when at least ``beta`` of its members share one decision and it has at
    least ``min_size`` members. With beta=1 and min_size=1 this is classical (crisp) rough set
    theory. The defaults suit real clinical data, which always contains some noise, and stop
    one-off unique patients from looking falsely certain.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, beta: float = 0.85, min_size: int = 3):
        self.X = X.reset_index(drop=True).astype(str)
        self.y = pd.Series(y).reset_index(drop=True).astype(str)
        self.attributes = list(self.X.columns)
        self.beta, self.min_size = beta, min_size
        self.reduct_: list | None = None

    # -- basic operations -------------------------------------------------------------------
    def _classes(self, attrs: list) -> pd.core.groupby.DataFrameGroupBy:
        return self.X.groupby(attrs if attrs else (lambda _: 0), sort=False)

    def _key(self, attrs: list) -> pd.Series:
        return self.X[attrs].agg("|".join, axis=1) if attrs else pd.Series("all", index=self.X.index)

    def _pure_mask(self, attrs: list) -> np.ndarray:
        """True for objects whose class is certain (majority share >= beta, size >= min_size)."""
        key = self._key(attrs)
        share = self.y.groupby(key).transform(lambda s: s.value_counts(normalize=True).iloc[0])
        size = self.y.groupby(key).transform("size")
        return ((share >= self.beta - 1e-12) & (size >= self.min_size)).to_numpy()

    def dependency(self, attrs: list) -> float:
        """Degree of dependency gamma: share of objects classified with certainty by ``attrs``."""
        return float(self._pure_mask(list(attrs)).mean())

    def positive_region(self, attrs: list | None = None) -> np.ndarray:
        return np.where(self._pure_mask(attrs or self.attributes))[0]

    def boundary_region(self, attrs: list | None = None) -> np.ndarray:
        return np.where(~self._pure_mask(attrs or self.attributes))[0]

    def approximations(self, decision: str, attrs: list | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper approximation of one decision class."""
        attrs = attrs or self.attributes
        key = self.X[attrs].agg("|".join, axis=1)
        is_d = self.y == decision
        all_d = is_d.groupby(key).transform("all")
        any_d = is_d.groupby(key).transform("any")
        return np.where(all_d)[0], np.where(any_d)[0]

    # -- data accuracy ----------------------------------------------------------------------
    def inconsistent_records(self, attrs: list | None = None) -> pd.DataFrame:
        """Groups of records with identical features but different labels.

        Returns one row per record in such groups, with the group's majority label and its
        share, so a reviewer can decide whether a label is wrong or the features are too coarse.
        """
        attrs = attrs or self.attributes
        key = self.X[attrs].agg("|".join, axis=1)
        bad = self.y.groupby(key).transform("nunique") > 1
        if not bad.any():
            return pd.DataFrame(columns=["row", "label", "majority_label", "majority_share", "group_size"])
        k, y = key[bad], self.y[bad]
        maj = y.groupby(k).agg(lambda s: s.value_counts().idxmax())
        share = y.groupby(k).agg(lambda s: s.value_counts(normalize=True).max())
        size = y.groupby(k).size()
        out = pd.DataFrame({"row": k.index, "label": y.values, "majority_label": k.map(maj).values,
                            "majority_share": k.map(share).values.round(3),
                            "group_size": k.map(size).values})
        out["disagrees_with_majority"] = out["label"] != out["majority_label"]
        return out.sort_values(["disagrees_with_majority", "majority_share", "group_size"],
                               ascending=[False, False, False]).reset_index(drop=True)

    def accuracy_filter(self, min_group: int = 3, min_majority: float = 0.75) -> np.ndarray:
        """Row positions to KEEP after removing likely mislabelled records.

        A record is dropped only when it disagrees with a clear majority (``min_majority``) in a
        group of at least ``min_group`` identical records. Smaller or evenly split groups are left
        alone, because there the disagreement is genuine uncertainty, not an error.
        """
        inc = self.inconsistent_records()
        if inc.empty:
            return np.arange(len(self.y))
        drop = inc[(inc["label"] != inc["majority_label"]) & (inc["group_size"] >= min_group)
                   & (inc["majority_share"] >= min_majority)]["row"].to_numpy()
        return np.setdiff1d(np.arange(len(self.y)), drop)

    # -- reducts and core -------------------------------------------------------------------
    def purity(self, attrs: list) -> float:
        """Average share of the majority decision within each class (a softer form of dependency).

        Used to break ties: with coarse bands, adding one feature often creates no fully pure
        group yet, so plain dependency stays flat. Purity still shows which feature helps most.
        """
        if not attrs:
            return float(self.y.value_counts(normalize=True).max())
        key = self.X[attrs].agg("|".join, axis=1)
        return float(self.y.groupby(key).agg(lambda s: s.value_counts().max()).sum() / len(self.y))

    def information(self, attrs: list) -> float:
        """Share of the uncertainty about the decision removed by ``attrs`` (0 to 1).

        Based on conditional entropy. Unlike variable-precision dependency it never falls when a
        feature is added, and it still moves when the outcome is rare, so it guides the search
        when dependency is flat. Groups smaller than ``min_size`` are pooled, so features that
        only split people into tiny groups (fitting noise) earn no credit.
        """
        def entropy(s: pd.Series) -> float:
            p = s.value_counts(normalize=True).to_numpy()
            return float(-(p * np.log2(p)).sum())
        base = entropy(self.y)
        if base == 0:
            return 1.0
        key = self._key(attrs)
        size = key.map(key.value_counts())
        key = key.where(size >= self.min_size, "__small__")
        cond = sum(len(g) / len(self.y) * entropy(g) for _, g in self.y.groupby(key))
        return 1 - cond / base

    def quick_reduct(self, tolerance: float = 0.01, min_gain: float = 0.005) -> list:
        """Greedy QuickReduct: add the feature that raises dependency most; information gain
        breaks ties.

        Stops when dependency reaches 1 - ``tolerance`` (crisp mode: that of the full feature set),
        or when no feature adds at least ``min_gain`` to dependency or to information (more
        features would only fit noise).
        """
        crisp = self.beta >= 1 and self.min_size <= 1
        target = (self.dependency(self.attributes) if crisp else 1.0) - tolerance
        chosen: list = []
        best, best_i = self.dependency(chosen), self.information(chosen)
        while best < target and len(chosen) < len(self.attributes):
            scores = {a: (self.dependency(chosen + [a]), self.information(chosen + [a]))
                      for a in self.attributes if a not in chosen}
            # Prefer a real gain in dependency; otherwise the feature with most information.
            a_best = max(scores, key=lambda a: (scores[a][0] - best >= min_gain, scores[a][1], scores[a][0]))
            g, i = scores[a_best]
            if g - best < min_gain and i - best_i < min_gain:
                break
            chosen.append(a_best)
            best, best_i = max(best, g), i
        # Remove any feature that is no longer needed once others were added.
        full_i = self.information(chosen)
        for a in list(chosen):
            rest = [c for c in chosen if c != a]
            if rest and self.dependency(rest) >= best - 1e-12 and self.information(rest) >= full_i - min_gain:
                chosen, full_i = rest, self.information(rest)
        self.reduct_ = chosen
        return chosen

    def core(self, attrs: list | None = None, tolerance: float = 0.01) -> list:
        """Features that cannot be removed without losing classifying power.

        Computed within ``attrs`` (by default the reduct, if one has been found). With many
        features almost every patient is unique, so every feature looks essential; working within
        the reduct and allowing a small ``tolerance`` gives a clinically meaningful core.
        """
        attrs = attrs or self.reduct_ or self.attributes
        full = self.dependency(attrs)
        return [a for a in attrs
                if self.dependency([c for c in attrs if c != a]) < full - tolerance]

    # -- rules ------------------------------------------------------------------------------
    def rules(self, attrs: list | None = None, min_support: int = 5, min_certainty: float = 0.0) -> list[Rule]:
        attrs = attrs or self.reduct_ or self.attributes
        key = self.X[attrs].agg("|".join, axis=1)
        class_sizes = self.y.value_counts()
        out = []
        for k, idx in key.groupby(key).groups.items():
            ys = self.y.loc[idx]
            if len(ys) < min_support:
                continue
            dec = ys.value_counts().idxmax()
            cert = (ys == dec).mean()
            if cert < min_certainty:
                continue
            cond = dict(zip(attrs, k.split("|")))
            out.append(Rule(cond, dec, float(cert), int(len(ys)), float((ys == dec).sum() / class_sizes[dec])))
        return sorted(out, key=lambda r: (-r.certainty, -r.support))

    def risk_groups(self, attrs: list | None = None, positive: str = "1", min_support: int = 20) -> pd.DataFrame:
        """For a yes/no outcome: each group of identical profiles with its rate of the outcome,
        highest first. Useful for screening, where the outcome is rare and few groups are 'pure'."""
        attrs = attrs or self.reduct_ or self.attributes
        key = self._key(attrs)
        g = (self.y == positive).groupby(key).agg(["mean", "size"])
        g = g[g["size"] >= min_support].sort_values("mean", ascending=False)
        prof = pd.DataFrame([dict(zip(attrs, k.split("|"))) for k in g.index], index=g.index)
        return prof.assign(rate=g["mean"].round(3), n=g["size"]).reset_index(drop=True)

    @staticmethod
    def suggested_beta(y) -> float:
        """A certainty threshold halfway between the most common outcome's share and 1.

        With a rare outcome (for example 15% of people), even "everyone at once" is 85% one way,
        so a fixed beta of 0.85 would call the whole population certain. This keeps beta above
        the base rate so a group counts as certain only if it is clearly purer than average.
        """
        base = pd.Series(y).value_counts(normalize=True).max()
        return round(float(base + (1 - base) / 2), 3)

    # -- triage of a new case ---------------------------------------------------------------
    def classify(self, x: dict, attrs: list | None = None) -> dict:
        """Where a new (already discretised) case falls.

        Returns status "certain" (at least ``beta`` of matching past cases share one label),
        "boundary" (matching cases disagree, so refer) or "unseen" (fewer than ``min_size``
        identical past cases; rely on the ML model and clinical judgement).
        """
        attrs = attrs or self.reduct_ or self.attributes
        mask = np.ones(len(self.X), dtype=bool)
        for a in attrs:
            mask &= (self.X[a] == str(x[a])).to_numpy()
        n = int(mask.sum())
        if n < self.min_size:
            return {"status": "unseen", "matches": n, "decisions": {}}
        dist = self.y[mask].value_counts(normalize=True).round(3).to_dict()
        top = max(dist.values())
        return {"status": "certain" if top >= self.beta else "boundary", "matches": n, "decisions": dist}


# Clinical cut points used to band the features before rough set analysis.
CLINICAL_CUTS = {
    **{f: SCORE_BANDS for f in ["Frequency_Score", "Urgency_Score", "Nocturia_Score", "Pain_Score", "QoL_Impact_Score"]},
    "Age": ([0, 40, 65, 200], ["<40", "40-64", "65+"]),
    "Duration_Months": ([-1, 1, 12, 1e9], ["<1m", "1-12m", ">12m"]),
    "PVR_ml": ([-1, 100, 200, 1e9], ["<100", "100-200", ">200"]),
    "Recurrent_UTI_Count": ([-1, 0, 2, 1e9], ["0", "1-2", "3+"]),
}

NHANES_CUTS = {
    "age": ([0, 39, 59, 74, 200], ["20-39", "40-59", "60-74", "75+"]),
    "bmi": ([0, 25, 30, 1e9], ["<25", "25-29", "30+"]),
    "nocturia": ([-1, 0, 1, 2, 10], ["0", "1", "2", "3+"]),
    "diabetes": ([-1, 0.25, 0.75, 2], ["no", "borderline", "yes"]),
}
