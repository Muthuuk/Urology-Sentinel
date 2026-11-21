# Urology-Sentinel (UroSentinel)

**Explainable, ethics-by-design AI for bladder health: population screening and clinical differential diagnosis.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muthuuk/Urology-Sentinel/blob/main/notebooks/UroSentinel_Colab.ipynb)

> **Research prototype only. Not a medical device. It does not diagnose any condition.** Anyone worried about bladder symptoms should speak to their GP or nurse.

UroSentinel is version 3 of *Urology prediction model 2*. It brings together two ABH 2026 studies:

- *A Comparative Deep Learning Study for Symptom-Based Overactive Bladder Prediction with SHAP Explainability* (Subbiah and Ramachandran)
- *Rough Set-Augmented Ensemble Machine Learning for OAB/BPS Differential Diagnosis* (Ramachandran and Kavitha)

## Run it

Click **Open in Colab** above, then choose **Runtime → Run all** (about 5 to 10 minutes). The notebook downloads the data, cleans it, trains and explains the models, and opens the interactive patient and clinician screens.

## What it does

| Part | What happens |
|---|---|
| **A. Population screening (US)** | Pools several NHANES cycles (a nationally representative US survey, about 25,000 adults with urology answers). Predicts an *OAB-wet proxy* (urge leakage at least a few times a month). The questions used to build the label are excluded from the inputs, which removes the feature-overlap limitation in the ABH 2026 paper. Survey weights keep results representative. |
| **B. International check (Poland)** | Classical rough set rules on the public UCI *Acute Inflammations* dataset (bladder inflammation and nephritis), which was originally built for rough set research. |
| **C. Clinical differential** | Six conditions (OAB, IC/BPS, acute cystitis, chronic or recurrent cystitis, neurogenic bladder, bladder outlet obstruction from BPH). Uses your own CSV in the Model 2 layout, or a clearly labelled synthetic cohort for demonstration. |
| **D. UK data** | A loader and external validation for UK Biobank, CPRD, OpenSAFELY or local NHS audit extracts, once access is approved. Nothing UK-specific is downloaded. |
| **E. Interactive screens** | A patient screen (plain language, likelihood band, reasons, NHS-style advice) and a clinician screen (probabilities, rough set triage, red flags, SHAP and LIME, report, decision log). |
| **F. Web app** | Optional Gradio app with a shareable link, for demonstrations and usability testing. |
| **G. Governance** | A V-EGRCSbyD checklist filled in from evidence, and model cards. |

## Methods in brief

- **Explainability:** SHAP (game-theory credit for each input) and LIME (a simple local model around one patient). Their agreement on the top reasons is reported as a trust check.
- **Rough set theory:** the variable-precision form (Ziarko), with a certainty threshold set above the base rate for rare outcomes. It is used to:
  - find contradictory records (data accuracy) and remove only clear minority labels, from training only;
  - find the reduct and core (explainable feature selection);
  - produce IF-THEN rules;
  - refer patients who fall in the boundary region instead of forcing a label.
- **Models:** logistic regression, random forest (Model 2), gradient boosting and a neural network (the ABH 2026 deep learning approach), compared with stratified cross-validation. For screening, the threshold is chosen to find at least 85% of true cases.
- **Ethics by design:** every cleaning step is logged with the principle it serves (privacy, accuracy, scope, transparency, justice). Protected characteristics (ethnicity, income) are used only to audit fairness, never as inputs. The fairness audit flags groups where the model misses more cases. Red-flag safety rules (for example NICE NG12 for visible haematuria at 45 or over) always override the model. Clinicians record their own decision, and it is logged.

## Data sources

| Source | Access | Notes |
|---|---|---|
| NHANES, CDC (US) | Public, downloaded automatically | Self-reported; cross-sectional |
| UCI Acute Inflammations (Poland) | Public, downloaded automatically | 120 patients; small |
| UK Biobank, CPRD, OpenSAFELY, NHS audit | **Approved access only** | Map to the harmonised columns (see `ds.UK_GUIDANCE`) |
| Your clinical CSV (Model 2 layout) | Yours | Must be anonymised, with a DPIA in place |
| Synthetic cohorts | Generated | For demonstration only; always labelled as synthetic |

If NHANES cannot be reached (for example on a locked-down network), the notebook falls back to a synthetic NHANES-shaped sample and says so clearly.

## Repository layout

```
urosentinel/      Python package
  data_sources.py   NHANES, UCI, UK loader, clinical cohort
  ethics.py         ethical cleaning, representativeness, fairness audit, V-EGRCSbyD checklist, model card
  roughsets.py      rough set theory: reducts, core, rules, boundary triage, inconsistency checks
  models.py         model comparison, training, thresholds, external validation
  explain.py        SHAP, LIME, agreement, plain-English summaries
  service.py        prediction services, red flags, guideline pointers, feedback log, HTML report
  app.py            patient and clinician screens (ipywidgets)
  gradio_app.py     shareable web app
notebooks/UroSentinel_Colab.ipynb
tests/            python -m pytest -q
```

## Before any clinical use

This is research software. Clinical use would need:

- validation on real, multicentre UK clinical data;
- a data protection impact assessment;
- clinical safety cases under DCB0129 and DCB0160;
- an assessment of its status as software as a medical device with the MHRA.

## Citation

Ramachandran, M. (2026). *UroSentinel: explainable, ethics-by-design urology prediction* (version 3.0). Forti5 Technologies Ltd. https://github.com/Muthuuk/Urology-Sentinel

MIT License.
