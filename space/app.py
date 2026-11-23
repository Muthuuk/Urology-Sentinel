"""UroSentinel web demo for Hugging Face Spaces.

On start-up the Space trains the two models exactly as the Colab notebook does, then opens the
patient and clinician screens as a Gradio web app with two tabs:

1. Screening model: NHANES survey data (United States). If the CDC website cannot be reached,
   a clearly labelled synthetic sample is used instead and the app says so.
2. Clinical model: the six-condition demonstration cohort (synthetic), checked with rough sets.

Research prototype only. Not a medical device. Do not enter real patient data.
"""

import warnings

warnings.filterwarnings("ignore")

# Hugging Face may give a new free Space "ZeroGPU" hardware, which refuses to start unless the app
# registers at least one GPU function. UroSentinel needs no graphics card (everything runs on the
# normal processor), so this small placeholder satisfies that check and is never used for the models.
try:
    import spaces
    _gpu = spaces.GPU
except ImportError:          # running on a laptop, Colab or CPU hardware
    def _gpu(func):
        return func


@_gpu
def gpu_placeholder():
    return "UroSentinel runs on the CPU"

from urosentinel import data_sources as ds, ethics, models as m, roughsets as rs
from urosentinel.gradio_app import build_app
from urosentinel.service import ClinicalService, ScreeningService

# ---- 1. Population screening model (logistic regression, the best model in the notebook) ----
# Three recent survey cycles keep start-up to a few minutes on the free hardware.
raw = ds.load_nhanes(["2015-2016", "2017-2018", "2021-2023"], cache_dir="/tmp/nhanes_cache")
adults = raw[raw["age"] >= 20]
clean, _ = ethics.ethical_clean(
    adults, valid_ranges={"age": (20, 120), "bmi": (12, 80), "nocturia": (0, 5)},
    id_cols=["seqn"], exclude_if={"pregnant": 1}, protected_features=ds.NHANES_AUDIT_ONLY)
clean = ds.add_nhanes_target(clean)
nh = clean.dropna(subset=[ds.NHANES_TARGET]).reset_index(drop=True)
screen_model = m.train_binary(nh, ds.NHANES_FEATURES, ds.NHANES_TARGET, model="logistic",
                              weight_col="survey_weight", target_sensitivity=0.85)
print("Screening model ready. Synthetic data:", raw.attrs.get("synthetic"), screen_model.test_metrics)

# ---- 2. Clinical six-condition model with rough set triage ----
clin_raw = ds.generate_clinical_cohort()
clin, _ = ethics.ethical_clean(
    clin_raw, id_cols=["Patient_ID"],
    valid_ranges={"Age": (18, 110), **{c: (0, 10) for c in ["Frequency_Score", "Urgency_Score", "Nocturia_Score",
                                                             "Pain_Score", "QoL_Impact_Score"]},
                  "PVR_ml": (0, 1500), "Duration_Months": (0, 600)})
feat = [c for c in ds.CLINICAL_FEATURES if clin[c].notna().any()]
R = rs.RoughSet(rs.discretise(clin[feat], cuts=rs.CLINICAL_CUTS), clin["Primary_Diagnosis"], beta=0.85, min_size=3)
keep = R.accuracy_filter(min_group=3, min_majority=0.75)     # leave likely mislabelled records out of training
clin_model = m.train_multiclass(clin, feat, "Primary_Diagnosis", keep_rows=keep)
print("Clinical model ready.", clin_model.test_metrics)

# ---- 3. Web app ----
demo = build_app(ScreeningService(screen_model), ClinicalService(clin_model, R, rs.CLINICAL_CUTS))

if __name__ == "__main__":
    demo.launch()
