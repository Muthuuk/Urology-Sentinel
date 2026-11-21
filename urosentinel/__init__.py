"""UroSentinel: explainable, ethics-by-design urology prediction (Model 2, version 3).

Builds on "Urology prediction model 2" (Random Forest bladder-condition predictor) and the
ABH 2026 work on NHANES-based overactive bladder (OAB) prediction with SHAP and on
rough set-augmented ensembles for OAB versus IC/BPS.

Modules
-------
data_sources : load NHANES (US), UCI Acute Inflammations (international), UK harmonised CSVs,
               and a calibrated synthetic clinical cohort for the six-condition differential.
ethics       : ethics-by-design data cleaning, data quality report, fairness audit, model card.
roughsets    : rough set theory (reducts, core, boundary region, rules, inconsistency checks).
models       : calibrated model training, evaluation and threshold tuning.
explain      : SHAP and LIME explanations, with plain-English summaries.
app          : interactive patient and clinician screens for Jupyter/Colab (ipywidgets).

This is research software. It is not a medical device and must not be used to diagnose
or treat patients.
"""

__version__ = "3.0.0"

DISCLAIMER = (
    "Research prototype only. Not a medical device. It does not diagnose any condition. "
    "Anyone worried about bladder symptoms should speak to their GP or nurse."
)
