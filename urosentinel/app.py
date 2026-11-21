"""Interactive screens for Jupyter and Google Colab (ipywidgets).

``patient_screen``: plain-language questions, a risk band, the main reasons and practical advice.
``clinician_screen``: full inputs, probabilities, rough set triage, red flags, SHAP and LIME,
a downloadable report, and a box to record the clinician's own decision.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import ipywidgets as w
from IPython.display import HTML, display

from . import DISCLAIMER
from .data_sources import CONDITIONS
from .service import ClinicalService, ScreeningService, html_report

BAND_COLOUR = {"low": "#1f7a4d", "moderate": "#b45309", "high": "#b42323"}


def patient_screen(service: ScreeningService) -> w.Widget:
    style = {"description_width": "260px"}
    lay = w.Layout(width="560px")
    age = w.IntSlider(55, 20, 90, description="Your age", style=style, layout=lay)
    sex = w.ToggleButtons(options=["Female", "Male"], description="Sex", style=style)
    height = w.IntSlider(165, 130, 210, description="Height (cm)", style=style, layout=lay)
    weight = w.IntSlider(75, 35, 200, description="Weight (kg)", style=style, layout=lay)
    diab = w.ToggleButtons(options=["No", "Borderline", "Yes"], description="Diabetes?", style=style)
    kid = w.Checkbox(description="Told by a doctor you have weak or failing kidneys", indent=False)
    stress = w.Checkbox(description="Leak when coughing, sneezing, lifting or exercising", indent=False)
    noct = w.IntSlider(1, 0, 5, description="Times you get up at night to pass urine (5 = 5+)", style=style, layout=lay)
    btn = w.Button(description="See my result", button_style="primary", icon="check")
    out = w.Output()

    def run(_):
        out.clear_output()
        with out:
            f = service.to_features(age.value, sex.value, height.value, weight.value, diab.value,
                                    kid.value, stress.value, noct.value)
            r = service.predict(f)
            c = BAND_COLOUR[r["band"]]
            reasons = "".join(f"<li>{t}</li>" for t in r["explanation"])
            advice = "".join(f"<li>{t}</li>" for t in r["advice"])
            display(HTML(f"""<div style="border-left:6px solid {c};padding:10px 14px;background:#f7f9fc;border-radius:8px">
            <h3 style="color:{c};margin:0">{r['band'].title()} likelihood</h3><p>{r['message']}</p>
            <p>Out of 100 people giving similar answers, about <b>{r['per_100']}</b> reported this kind of leak.</p>
            <b>What affected your result most</b><ul>{reasons}</ul><b>What you can do</b><ul>{advice}</ul>
            <p style="font-size:12px;color:#555">{r['disclaimer']}</p></div>"""))

    btn.on_click(run)
    return w.VBox([w.HTML("<h3>Bladder health check</h3><p>Answer a few questions. Nothing you enter is stored or sent anywhere.</p>"),
                   age, sex, height, weight, diab, kid, stress, noct, btn, out])


def clinician_screen(service: ClinicalService) -> w.Widget:
    style = {"description_width": "190px"}
    sl = lambda d, v, lo, hi: w.IntSlider(v, lo, hi, description=d, style=style, layout=w.Layout(width="440px"))
    fields = {
        "Age": sl("Age", 58, 18, 95), "Female": w.ToggleButtons(options=[("Female", 1), ("Male", 0)], description="Sex", style=style),
        "Frequency_Score": sl("Frequency (0-10)", 7, 0, 10), "Urgency_Score": sl("Urgency (0-10)", 8, 0, 10),
        "Nocturia_Score": sl("Nocturia (0-10)", 6, 0, 10), "Pain_Score": sl("Bladder pain (0-10)", 2, 0, 10),
        "QoL_Impact_Score": sl("QoL impact (0-10)", 6, 0, 10),
        "Duration_Months": w.FloatText(24, description="Duration (months)", style=style),
        "PVR_ml": w.FloatText(40, description="Post-void residual (ml)", style=style),
        "Recurrent_UTI_Count": sl("UTIs in past year", 0, 0, 12),
    }
    flags = {k: w.Checkbox(description=d, indent=False) for k, d in [
        ("Fever", "Fever"), ("Dysuria", "Dysuria"), ("Haematuria", "Visible haematuria"),
        ("Nitrite_Positive", "Nitrite positive"), ("Neuro_History", "Neurological condition"), ("Weak_Stream", "Weak stream")]}
    btn = w.Button(description="Assess", button_style="primary", icon="stethoscope")
    out = w.Output()
    decision = w.Dropdown(options=CONDITIONS + ["Other / undecided"], description="Your decision", style=style)
    comment = w.Text(description="Comment", style=style, layout=w.Layout(width="560px"))
    save = w.Button(description="Record my decision", icon="save")
    saved = w.Output()
    state = {}

    def inputs():
        x = {k: float(v.value) for k, v in fields.items()}
        x.update({k: float(v.value) for k, v in flags.items()})
        return x

    def run(_):
        out.clear_output()
        with out:
            x = inputs()
            r = service.predict(x)
            state.update(x=x, r=r)
            decision.value = r["suggestion"]
            flag_html = "".join(f"<li style='color:#b42323'><b>{f}</b></li>" for f in r["red_flags"])
            rst = r["rough_set"]
            display(HTML(f"""<div style="padding:10px;background:#f7f9fc;border-radius:8px">
            {'<b>RED FLAGS</b><ul>' + flag_html + '</ul>' if flag_html else ''}
            <h3 style="margin:4px 0">Suggestion: {r['suggestion']} ({r['probabilities'][r['suggestion']]:.0%})</h3>
            <p><b>Triage:</b> {r['triage']}<br><b>Rough set:</b> {rst['status']} ({rst['matches']} similar past patients)
            {('<br><b>Rule:</b> ' + r['matched_rule']) if r['matched_rule'] else ''}</p>
            <p><b>Guideline pointer:</b> {r['guideline']}</p>
            <p><b>SHAP and LIME agree on {r['agreement']:.0%} of the top 3 reasons.</b></p></div>"""))
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
            names, vals = list(r["probabilities"])[::-1], list(r["probabilities"].values())[::-1]
            ax[0].barh(names, vals, color="#0B6E99"); ax[0].set_xlim(0, 1); ax[0].set_title("Probability by condition")
            lime = r["lime"][::-1]
            ax[1].barh([c for c, _ in lime], [v for _, v in lime], color=["#1f7a4d" if v > 0 else "#b42323" for _, v in lime])
            ax[1].set_title(f"LIME: reasons for {r['suggestion']}")
            plt.tight_layout(); plt.show()
            import shap
            shap.plots.waterfall(r["shap"][0], max_display=8, show=False)
            plt.title(f"SHAP: how each input moved the score for {r['suggestion']}"); plt.show()
            rep = "urosentinel_report.html"
            open(rep, "w").write(html_report(x, r))
            print(f"Report saved as {rep}. In Colab: from google.colab import files; files.download('{rep}')")

    def record(_):
        saved.clear_output()
        with saved:
            if "r" not in state:
                print("Run an assessment first.")
                return
            p = service.log_feedback(state["x"], state["r"], decision.value, comment.value)
            print(f"Recorded in {p}. Your decision is final; the tool only suggests.")

    btn.on_click(run)
    save.on_click(record)
    return w.VBox([w.HTML(f"<h3>Clinician decision support</h3><p style='font-size:12px'>{DISCLAIMER}</p>"),
                   w.HBox([w.VBox(list(fields.values())), w.VBox(list(flags.values()))]), btn, out,
                   w.HTML("<b>Human oversight</b>: record whether you agree."), decision, comment, save, saved])
