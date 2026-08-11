"""Fleet maintenance monitoring dashboard -- phase 3 of the capstone.

Calls src/classifier.py directly; this file is UI only, no model logic lives here.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
from classifier import classify_fault, classify_fault_batch, explain_prediction, FLAG_THRESHOLD  # noqa: E402

SCENARIOS_DIR = BASE_DIR / "data" / "processed" / "scenarios"
VEHICLE_START_TIME = "2020-01-01 00:00:00"  # real service-start date for all 4 vehicles in this dataset

SCENARIOS = {
    "Normal driving": {
        "file": "normal.csv",
        "blurb": "A calm 48h window, no fault episode nearby.",
    },
    "Gradual degradation": {
        "file": "escalating.csv",
        "blurb": "48h window ending ~12h before an actual fault episode in the data -- risk climbing, not yet flagged.",
    },
    "Pre-fault escalation": {
        "file": "prefault.csv",
        "blurb": "48h window ending right at a real fault episode in the data.",
    },
}

st.set_page_config(page_title="Fleet maintenance monitoring", layout="wide")
st.title("Fleet maintenance monitoring")
st.caption(
    "Vehicle Fault Diagnosis & Repair Copilot -- XGBoost classifier, F2 = 0.594 on the held-out test set. "
    "Team NodePair."
)

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    vehicle = st.selectbox("Vehicle / usage profile", ["Heavy user", "Daily user", "Moderate user", "Rare user"])
with col_b:
    scenario_name = st.selectbox("Scenario (pre-loaded 48h window)", list(SCENARIOS.keys()), index=2)
with col_c:
    st.write("")
    run = st.button("Run prediction", type="primary")

st.caption(SCENARIOS[scenario_name]["blurb"])
if vehicle != "Heavy user":
    st.info(
        "These three scenarios were pulled from real heavy_user data. Showing that vehicle's "
        "readings regardless of the profile selected above until scenarios exist for the others."
    )

if run or "last_scenario" not in st.session_state or st.session_state.get("last_scenario") != scenario_name:
    st.session_state["last_scenario"] = scenario_name

    scenario_path = SCENARIOS_DIR / SCENARIOS[scenario_name]["file"]
    readings = pd.read_csv(scenario_path, parse_dates=["timestamp"])

    result = classify_fault(readings, vehicle_start_time=VEHICLE_START_TIME)
    trend = classify_fault_batch(readings, vehicle_start_time=VEHICLE_START_TIME)
    factors = explain_prediction(readings, vehicle_start_time=VEHICLE_START_TIME)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Risk status", result["risk_level"])
    with m2:
        st.metric("Probability of fault (next 6h)", f"{result['probability']:.0%}")
    with m3:
        st.metric("Decision threshold", f"{FLAG_THRESHOLD:.0%}")

    st.subheader("Probability over the window")
    chart_df = trend.set_index("timestamp")[["probability"]]
    st.line_chart(chart_df)

    st.subheader("Why: top contributing readings")
    st.caption(
        "From the model's own feature contributions (XGBoost's built-in SHAP values) -- not RAG, "
        "no documents involved. Positive impact pushed the probability up, negative pushed it down."
    )
    for f in factors:
        direction = "pushed risk up" if f["impact"] > 0 else "pushed risk down"
        st.write(f"**{f['feature']}** = {f['value']:g} -- {direction} ({f['impact']:+.3f})")

    with st.expander("Raw readings for this window"):
        st.dataframe(readings, use_container_width=True)

st.divider()
st.caption(
    "Model: models/xgboost_all_vehicles.joblib, 53 selected features, trained on the full "
    "140,067-row training set (no subsampling). See docs/Business_Goal_and_Process.md and "
    "docs/Classifier_and_Dashboard_FAQ.md for the full writeup."
)
