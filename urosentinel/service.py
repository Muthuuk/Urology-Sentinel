"""Prediction services shared by the notebook screens, the Gradio web app and the tests.

Two services:
  * ``ScreeningService``: population screening for urgency urinary incontinence (OAB-wet proxy),
    trained on NHANES. Written for patients: plain language, risk bands, no diagnosis.
  * ``ClinicalService``: six-condition differential for clinicians, with rough set triage,
    red-flag checks, guideline pointers, SHAP and LIME, and a feedback log so the clinician's
    own decision is always recorded (human oversight).
"""

from __future__ import annotations

import csv
import datetime as _dt
import html
import os

import numpy as np
import pandas as pd

from . import DISCLAIMER, __version__
from . import explain as ex
from .roughsets import discretise

# --------------------------------------------------------------------------------------------
# Population screening (patient facing)
# --------------------------------------------------------------------------------------------

PATIENT_ADVICE = [
    "Keep a simple bladder diary for 3 days (what you drink, when you pass urine, any leaks).",
    "Cutting down on caffeine, fizzy drinks and alcohol helps many people.",
    "Pelvic floor exercises help with leaks; the NHS Squeezy app can guide you.",
    "Speak to your GP or practice nurse if symptoms bother you. Help is available and it works.",
]


class ScreeningService:
    def __init__(self, trained_model, moderate_cut: float | None = None, high_cut: float = 0.5):
        self.tm = trained_model
        self.low_cut = trained_model.threshold
        self.high_cut = max(high_cut, self.low_cut + 0.05) if moderate_cut is None else moderate_cut

    @staticmethod
    def to_features(age, sex, height_cm, weight_kg, diabetes, kidney, stress_leak, nocturia) -> dict:
        bmi = weight_kg / (height_cm / 100) ** 2 if height_cm and weight_kg else np.nan
        return {"age": float(age), "female": 1.0 if str(sex).lower().startswith("f") else 0.0,
                "bmi": round(float(bmi), 1), "diabetes": {"No": 0.0, "Borderline": 0.5, "Yes": 1.0}.get(diabetes, 0.0),
                "kidney_weak": float(bool(kidney)), "stress_leak": float(bool(stress_leak)),
                "nocturia": float(nocturia)}

    def predict(self, features: dict, explain: bool = True) -> dict:
        row = pd.DataFrame([features])[self.tm.features]
        p = float(self.tm.predict_proba(row)[0, 1])
        band = "low" if p < self.low_cut else ("moderate" if p < self.high_cut else "high")
        out = {"probability": p, "band": band, "per_100": int(round(p * 100)), "features": features}
        if explain:
            e = ex.shap_explain(self.tm, row)
            top = ex.top_features_shap(e, k=3)
            out["shap_top"] = top
            out["explanation"] = ex.plain_english(top, features, "patient")
            out["shap"] = e
        out["message"] = {
            "low": "Your answers suggest a low chance of bladder urgency with leaks.",
            "moderate": "Your answers suggest a moderate chance of bladder urgency with leaks.",
            "high": "Your answers suggest a higher chance of bladder urgency with leaks.",
        }[band]
        out["advice"] = PATIENT_ADVICE if band != "low" else PATIENT_ADVICE[-1:]
        out["disclaimer"] = DISCLAIMER
        return out


# --------------------------------------------------------------------------------------------
# Clinical differential (clinician facing)
# --------------------------------------------------------------------------------------------

GUIDELINE_POINTERS = {
    "OAB": "Overactive bladder: lifestyle advice and bladder training first; drug options (antimuscarinic "
           "or mirabegron) if needed, with attention to anticholinergic burden in older people. "
           "See NICE NG123 (women) and CG97 (men).",
    "IC_BPS": "Interstitial cystitis / bladder pain syndrome: exclude infection and other causes; "
              "multimodal care (diet, pain management, pelvic floor physiotherapy); consider specialist "
              "referral. See EAU and AUA guidance.",
    "Acute_Cystitis": "Acute cystitis: confirm with dipstick and culture where indicated; antimicrobial choice "
                      "per local guidance. See NICE NG109.",
    "Chronic_Cystitis": "Recurrent or chronic infection: confirm infections with culture before treating; "
                        "look for underlying causes. See NICE NG112.",
    "Neurogenic_Bladder": "Neurogenic bladder: assess bladder emptying and the upper urinary tract; refer to "
                          "neuro-urology.",
    "BOO_BPH": "Bladder outlet obstruction / benign prostatic enlargement: symptom score (IPSS), PSA counselling, "
               "assess emptying; refer if retention. See NICE CG97.",
}


def red_flags(x: dict) -> list[str]:
    """Safety checks that override any model output."""
    flags = []
    if x.get("Haematuria") == 1 and x.get("Age", 0) >= 45:
        flags.append("Visible haematuria at age 45 or over: consider the urgent suspected cancer pathway (NICE NG12).")
    if x.get("Fever") == 1 and (x.get("Dysuria") == 1 or x.get("Nitrite_Positive") == 1):
        flags.append("Fever with urinary symptoms: assess for upper urinary tract infection or sepsis.")
    if (x.get("PVR_ml") or 0) >= 300:
        flags.append("Post-void residual of 300 ml or more: assess for chronic retention and check kidney function.")
    if x.get("Neuro_History") == 1 and (x.get("PVR_ml") or 0) >= 200:
        flags.append("Neurological history with poor emptying: refer to neuro-urology.")
    return flags


class ClinicalService:
    def __init__(self, trained_model, rough_set, cuts: dict, feedback_path: str = "clinician_feedback.csv"):
        self.tm, self.rs, self.cuts, self.feedback_path = trained_model, rough_set, cuts, feedback_path

    def predict(self, x: dict, explain: bool = True) -> dict:
        row = pd.DataFrame([x])[self.tm.features].astype(float)
        prob = self.tm.predict_proba(row)[0]
        order = np.argsort(-prob)
        probs = {self.tm.classes[i]: float(prob[i]) for i in order}
        top_i = int(order[0])
        top = self.tm.classes[top_i]
        xd = discretise(row, cuts=self.cuts).iloc[0].to_dict()
        rst = self.rs.classify(xd)
        rules = [r for r in self.rs.rules(min_support=self.rs.min_size)
                 if all(xd.get(k) == v for k, v in r.conditions.items())]
        out = {"probabilities": probs, "suggestion": top, "rough_set": rst,
               "matched_rule": str(rules[0]) if rules else None,
               "red_flags": red_flags(x), "guideline": GUIDELINE_POINTERS.get(top, "")}
        # Triage: refer if rough sets put the patient in the boundary, or the model is unsure.
        margin = prob[order[0]] - prob[order[1]]
        out["refer"] = rst["status"] == "boundary" or margin < 0.2
        out["triage"] = ("Refer for specialist review: similar past patients had different diagnoses."
                         if rst["status"] == "boundary" else
                         "Refer for review: the model is unsure between the top two conditions."
                         if margin < 0.2 else
                         "Consistent with similar past patients." if rst["status"] == "certain" else
                         "No closely matching past patients; rely on clinical judgement.")
        if explain:
            e = ex.shap_explain(self.tm, row, class_index=top_i)
            top_s = ex.top_features_shap(e, k=4)
            lime_list, _ = ex.lime_explain(self.tm, row, class_index=top_i, num_features=6)
            out.update({"shap": e, "shap_top": top_s, "lime": lime_list,
                        "agreement": ex.agreement(top_s, lime_list, self.tm.features, k=3),
                        "explanation": ex.plain_english(top_s, x, "clinician")})
        return out

    def log_feedback(self, x: dict, result: dict, clinician_decision: str, comment: str = "",
                     clinician_id: str = "anonymous") -> str:
        """Record what the clinician decided. The clinician always has the final say."""
        new = not os.path.exists(self.feedback_path)
        with open(self.feedback_path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["timestamp", "model_version", "clinician_id", "suggestion", "clinician_decision",
                            "agreed", "rough_set_status", "refer", "comment", "inputs"])
            w.writerow([_dt.datetime.now().isoformat(timespec="seconds"), __version__, clinician_id,
                        result["suggestion"], clinician_decision, clinician_decision == result["suggestion"],
                        result["rough_set"]["status"], result["refer"], comment, repr(x)])
        return self.feedback_path


def html_report(x: dict, result: dict) -> str:
    """A one-page HTML report a clinician can save or print."""
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.1%}</td></tr>" for k, v in result["probabilities"].items())
    inputs = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in x.items())
    flags = "".join(f"<li>{html.escape(f)}</li>" for f in result["red_flags"]) or "<li>None</li>"
    expl = "".join(f"<li>{html.escape(t)}</li>" for t in result.get("explanation", []))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>UroSentinel report</title>
<style>body{{font-family:Arial,sans-serif;max-width:800px;margin:20px auto}}td{{border-bottom:1px solid #ddd;padding:4px 8px}}
.warn{{background:#fff3cd;padding:8px;border-radius:6px}}</style></head><body>
<h2>UroSentinel decision support report</h2><p class="warn">{html.escape(DISCLAIMER)}</p>
<p>Generated {_dt.datetime.now():%d %b %Y %H:%M} · model version {__version__}</p>
<h3>Suggestion: {html.escape(result['suggestion'])}</h3><p>{html.escape(result['triage'])}</p>
<h3>Red flags</h3><ul>{flags}</ul>
<h3>Probabilities</h3><table>{rows}</table>
<h3>Main reasons (SHAP)</h3><ul>{expl}</ul>
<p>Rough set rule: {html.escape(result['matched_rule'] or 'no matching rule')}</p>
<h3>Guideline pointer</h3><p>{html.escape(result['guideline'])}</p>
<h3>Inputs</h3><table>{inputs}</table></body></html>"""
