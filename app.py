"""Credit Risk Assessment with Explainable ML - Streamlit app."""
import html

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Credit Risk Assessment", layout="wide")


@st.cache_resource
def load_everything():
    bundle = joblib.load("credit_model.pkl")
    return bundle, shap.TreeExplainer(bundle["model"])


bundle, explainer = load_everything()
final_model = bundle["model"]
FEATURE_COLUMNS = bundle["columns"]
label_maps = bundle["label_maps"]
template = bundle["template"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

LOW_CUTOFF, HIGH_CUTOFF = 0.35, 0.60     # risk bands: edit after you see real probabilities

def pretty(name):
    for c in cat_cols:
        if name.startswith(c + "_"):
            return f"{c.replace('_', ' ')}: {name[len(c) + 1:]}"
    return name.replace("_", " ")

def describe_feature(name, value):
    if name not in num_cols:
        return pretty(name) if value == 1 else "NOT " + pretty(name)
    return f"{pretty(name)} = {value:g}"

def explain_in_words(sv, top=3):
    s = pd.Series(sv.values, index=sv.feature_names)
    vals = dict(zip(sv.feature_names, sv.data))
    up = s[s > 0].sort_values(ascending=False).head(top)
    down = s[s < 0].sort_values().head(top)
    return ([describe_feature(n, vals[n]) for n in up.index],
            [describe_feature(n, vals[n]) for n in down.index])

CSS = """
<style>
.cr-head h1 {font-size: 1.7rem; margin: 0 0 .25rem 0; letter-spacing: -0.01em;}
.cr-head p {margin: 0; opacity: .75; max-width: 60ch;}
.cr-card {background:#ffffff; color:#1b2433; border:1px solid #d9dee7; border-radius:10px; padding:20px 22px; font-family:inherit;}
.cr-top {display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;}
.cr-badge {display:inline-block; padding:5px 14px; border-radius:999px; font-weight:600; font-size:.95rem; color:#fff;}
.cr-prob {font-size:2.6rem; font-weight:700; line-height:1; letter-spacing:-0.02em;}
.cr-sub {font-size:.9rem; color:#5b6575; margin-top:4px;}
.cr-bar {position:relative; display:flex; height:10px; border-radius:6px; overflow:hidden; margin:18px 0 6px 0;}
.cr-bar span {display:block; height:100%;}
.cr-mark {position:absolute; top:-5px; width:4px; height:20px; background:#1b2433; border-radius:2px; transform:translateX(-2px);}
.cr-scale {display:flex; justify-content:space-between; font-size:.75rem; color:#5b6575;}
.cr-cols {display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:20px;}
@media (max-width:640px){ .cr-cols {grid-template-columns:1fr;} }
.cr-cols h4 {margin:0 0 8px 0; font-size:.95rem;}
.cr-cols ul {margin:0; padding-left:18px;}
.cr-cols li {margin-bottom:5px; font-size:.92rem; line-height:1.35;}
.cr-note {margin-top:18px; padding-top:12px; border-top:1px solid #e6e9ef; font-size:.8rem; color:#5b6575;}
.cr-empty {text-align:center; color:#5b6575; padding:36px 16px;}
</style>
"""

LEVELS = {
    "LOW":    ("Low risk",    "#2f7d5b"),
    "MEDIUM": ("Medium risk", "#b7791f"),
    "HIGH":   ("High risk",   "#b3402f"),
}

def risk_level(prob):
    if prob < LOW_CUTOFF:
        return "LOW"
    if prob < HIGH_CUTOFF:
        return "MEDIUM"
    return "HIGH"

def li(items, empty):
    if not items:
        return f"<li>{empty}</li>"
    return "".join(f"<li>{html.escape(i)}</li>" for i in items)

def result_card(prob, up, down):
    label, color = LEVELS[risk_level(prob)]
    low, mid = LOW_CUTOFF * 100, (HIGH_CUTOFF - LOW_CUTOFF) * 100
    high = 100 - low - mid
    return f"""
<div class="cr-card">
  <div class="cr-top">
    <div>
      <div class="cr-prob">{prob:.0%}</div>
      <div class="cr-sub">estimated chance of default</div>
    </div>
    <span class="cr-badge" style="background:{color}">{label}</span>
  </div>
  <div class="cr-bar">
    <span style="width:{low:.1f}%; background:#cfe7da"></span>
    <span style="width:{mid:.1f}%; background:#f3e2bd"></span>
    <span style="width:{high:.1f}%; background:#f0c9c2"></span>
    <div class="cr-mark" style="left:{prob*100:.1f}%"></div>
  </div>
  <div class="cr-scale"><span>0%</span><span>{LOW_CUTOFF:.0%}</span><span>{HIGH_CUTOFF:.0%}</span><span>100%</span></div>
  <div class="cr-cols">
    <div><h4 style="color:#b3402f">What raises the risk</h4><ul>{li(up, "Nothing stands out")}</ul></div>
    <div><h4 style="color:#2f7d5b">What lowers the risk</h4><ul>{li(down, "Nothing stands out")}</ul></div>
  </div>
  <div class="cr-note">Educational project trained on 1000 historical German loan records. Not for real lending decisions.</div>
</div>"""



def flat(s):
    """Remove indentation so Streamlit's markdown never treats the card as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def choices(col):
    return list(label_maps[col].values())


DEFAULTS = {
    "amount": 3000, "duration": 24, "purpose": "radio or TV", "installment": 3,
    "checking": "0 to 200 DM", "savings": "under 100 DM",
    "history": "existing credits paid duly till now",
    "age": 35, "employment": "1 to 4 years", "housing": "own", "job": "skilled employee",
}
SAFE = dict(amount=1500, duration=12, purpose="furniture or equipment", installment=2,
            checking="200+ DM", savings="1000+ DM",
            history="existing credits paid duly till now",
            age=45, employment="7+ years", housing="own", job="skilled employee")
RISKY = dict(amount=12000, duration=48, purpose="new car", installment=4,
             checking="under 0 DM", savings="under 100 DM",
             history="critical account or credits elsewhere",
             age=22, employment="under 1 year", housing="rent", job="unskilled resident")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("assessed", False)


def load_profile(profile):
    for k, v in profile.items():
        st.session_state[k] = v
    st.session_state["assessed"] = True


def run_assess():
    st.session_state["assessed"] = True


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(flat("""<div class="cr-head"><h1>Credit risk assessment</h1>
<p>Enter an applicant's details. The model estimates the chance of default and shows which factors drove the result.</p></div>"""),
            unsafe_allow_html=True)
st.write("")

left, right = st.columns([5, 6], gap="large")

with left:
    st.subheader("Loan")
    st.slider("Credit amount (DM)", 250, 20000, step=50, key="amount")
    st.slider("Loan duration (months)", 4, 72, step=1, key="duration")
    st.selectbox("Purpose of loan", choices("purpose"), key="purpose")
    st.slider("Instalment rate (1 to 4, % of income)", 1, 4, step=1, key="installment")

    st.subheader("Finances")
    st.selectbox("Checking account", choices("checking_status"), key="checking")
    st.selectbox("Savings", choices("savings"), key="savings")
    st.selectbox("Credit history", choices("credit_history"), key="history")

    st.subheader("Applicant")
    st.slider("Age", 19, 75, step=1, key="age")
    st.selectbox("Employed for", choices("employment_since"), key="employment")
    st.selectbox("Housing", choices("housing"), key="housing")
    st.selectbox("Job type", choices("job"), key="job")

    st.button("Assess risk", type="primary", on_click=run_assess)
    b1, b2 = st.columns(2)
    b1.button("Try a low-risk profile", on_click=load_profile, args=(SAFE,))
    b2.button("Try a high-risk profile", on_click=load_profile, args=(RISKY,))

with right:
    if not st.session_state["assessed"]:
        st.markdown(flat('<div class="cr-card cr-empty">Fill in the applicant details and select <b>Assess risk</b>.</div>'),
                    unsafe_allow_html=True)
    else:
        s = st.session_state
        row = dict(template)
        row.update({
            "checking_status": s["checking"], "duration_months": s["duration"],
            "credit_history": s["history"], "purpose": s["purpose"],
            "credit_amount": s["amount"], "savings": s["savings"],
            "employment_since": s["employment"], "installment_rate": s["installment"],
            "age": s["age"], "housing": s["housing"], "job": s["job"],
        })
        row_enc = pd.get_dummies(pd.DataFrame([row]), dtype=int).reindex(
            columns=FEATURE_COLUMNS, fill_value=0)

        prob = float(final_model.predict_proba(row_enc)[0, 1])
        sv = explainer(row_enc)[:, :, 1][0]
        up, down = explain_in_words(sv, top=3)

        st.markdown(flat(result_card(prob, up, down)), unsafe_allow_html=True)
        st.write("")
        st.markdown("**How each factor moved the prediction (SHAP)**")
        plt.close("all")
        shap.plots.waterfall(sv, max_display=10, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
