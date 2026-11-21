"""A shareable web app (Gradio) built on the same services as the notebook screens.

In Colab: ``demo = build_app(screening, clinical); demo.launch(share=True)`` gives a temporary
public link (about 72 hours) that colleagues can open in a browser, which is useful for
demonstrations and usability testing. Do not enter real patient data into a shared link.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import DISCLAIMER
from .service import ClinicalService, ScreeningService


def build_app(screening: ScreeningService, clinical: ClinicalService):
    import gradio as gr

    def patient(age, sex, height, weight, diabetes, kidney, stress, nocturia):
        f = screening.to_features(age, sex, height, weight, diabetes, kidney, stress, nocturia)
        r = screening.predict(f)
        text = (f"### {r['band'].title()} likelihood\n{r['message']}\n\n"
                f"Out of 100 people giving similar answers, about **{r['per_100']}** reported this kind of leak.\n\n"
                "**What affected your result most**\n" + "\n".join(f"- {t}" for t in r["explanation"]) +
                "\n\n**What you can do**\n" + "\n".join(f"- {t}" for t in r["advice"]) + f"\n\n_{DISCLAIMER}_")
        return text

    def clinician(age, female, freq, urg, noct, pain, qol, dur, pvr, uti, fever, dysuria, haem, nitrite, neuro, weak):
        x = {"Age": age, "Female": 1.0 if female == "Female" else 0.0, "Frequency_Score": freq, "Urgency_Score": urg,
             "Nocturia_Score": noct, "Pain_Score": pain, "QoL_Impact_Score": qol, "Duration_Months": dur,
             "PVR_ml": pvr, "Recurrent_UTI_Count": uti, "Fever": float(fever), "Dysuria": float(dysuria),
             "Haematuria": float(haem), "Nitrite_Positive": float(nitrite), "Neuro_History": float(neuro),
             "Weak_Stream": float(weak)}
        r = clinical.predict(x)
        md = ("".join(f"**RED FLAG:** {f}\n\n" for f in r["red_flags"]) +
              f"### Suggestion: {r['suggestion']}\n**Triage:** {r['triage']}\n\n"
              f"**Rough set:** {r['rough_set']['status']} ({r['rough_set']['matches']} similar past patients)\n\n"
              f"**Guideline pointer:** {r['guideline']}\n\n"
              f"**SHAP and LIME agree on {r['agreement']:.0%} of the top 3 reasons.**\n\n" +
              "\n".join(f"- {t}" for t in r["explanation"]) + f"\n\n_{DISCLAIMER}_")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(list(r["probabilities"])[::-1], list(r["probabilities"].values())[::-1], color="#0B6E99")
        ax.set_xlim(0, 1); ax.set_title("Probability by condition"); plt.tight_layout()
        return md, fig

    with gr.Blocks(title="UroSentinel") as demo:
        gr.Markdown(f"# UroSentinel: explainable bladder health support\n_{DISCLAIMER}_")
        with gr.Tab("Patient check"):
            inp = [gr.Slider(20, 90, 55, step=1, label="Age"), gr.Radio(["Female", "Male"], value="Female", label="Sex"),
                   gr.Slider(130, 210, 165, step=1, label="Height (cm)"), gr.Slider(35, 200, 75, step=1, label="Weight (kg)"),
                   gr.Radio(["No", "Borderline", "Yes"], value="No", label="Diabetes"),
                   gr.Checkbox(label="Weak or failing kidneys"), gr.Checkbox(label="Leak when coughing, sneezing or lifting"),
                   gr.Slider(0, 5, 1, step=1, label="Times up at night to pass urine (5 = 5+)")]
            gr.Button("See my result", variant="primary").click(patient, inp, gr.Markdown())
        with gr.Tab("Clinician"):
            s = lambda l, v: gr.Slider(0, 10, v, step=1, label=l)
            inp2 = [gr.Slider(18, 95, 58, step=1, label="Age"), gr.Radio(["Female", "Male"], value="Female", label="Sex"),
                    s("Frequency", 7), s("Urgency", 8), s("Nocturia", 6), s("Bladder pain", 2), s("QoL impact", 6),
                    gr.Number(24, label="Duration (months)"), gr.Number(40, label="Post-void residual (ml)"),
                    gr.Slider(0, 12, 0, step=1, label="UTIs in past year"),
                    gr.Checkbox(label="Fever"), gr.Checkbox(label="Dysuria"), gr.Checkbox(label="Visible haematuria"),
                    gr.Checkbox(label="Nitrite positive"), gr.Checkbox(label="Neurological condition"), gr.Checkbox(label="Weak stream")]
            gr.Button("Assess", variant="primary").click(clinician, inp2, [gr.Markdown(), gr.Plot()])
    return demo
