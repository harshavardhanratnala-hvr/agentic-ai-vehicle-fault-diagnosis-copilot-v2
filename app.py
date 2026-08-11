"""Fleet maintenance monitoring dashboard -- phase 3 of the capstone.

Calls src/classifier.py directly; this file is UI only, no model logic lives here.
Run with: streamlit run app.py
"""

import math
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
        "gauge_fraction": 0.15,
        "gauge_color": "#22c55e",
    },
    "Gradual degradation": {
        "file": "escalating.csv",
        "blurb": "48h window ending ~12h before an actual fault episode in the data -- risk climbing, not yet flagged.",
        "gauge_fraction": 0.55,
        "gauge_color": "#f59e0b",
    },
    "Pre-fault escalation": {
        "file": "prefault.csv",
        "blurb": "48h window ending right at a real fault episode in the data.",
        "gauge_fraction": 0.92,
        "gauge_color": "#ef4444",
    },
}

# Usage intensity, low to high -- drives the gauge fill on each profile's icon.
VEHICLE_PROFILES = {
    "Rare user": {"gauge_fraction": 0.15, "gauge_color": "#22c55e"},
    "Moderate user": {"gauge_fraction": 0.45, "gauge_color": "#eab308"},
    "Daily user": {"gauge_fraction": 0.7, "gauge_color": "#f97316"},
    "Heavy user": {"gauge_fraction": 0.92, "gauge_color": "#ef4444"},
}


def _car_gauge_icon_svg(fraction: float, color: str) -> str:
    """A small car-with-gauge icon; the gauge arc fills to `fraction` in `color`.

    Streamlit's st.radio has no per-option image slot, so this icon is paired with a
    plain button acting as the visible "radio button" instead (see render_image_choice).
    """
    cx, cy, r = 50, 40, 30
    angle = math.radians(180 - fraction * 180)
    nx, ny = cx + (r - 6) * math.cos(angle), cy - (r - 6) * math.sin(angle)
    arc_len = math.pi * r
    return (
        '<svg viewBox="0 0 100 100" width="72" height="72" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="#9aa0a6" '
        'stroke-width="7" stroke-linecap="round" opacity="0.3"/>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="{color}" '
        f'stroke-width="7" stroke-linecap="round" stroke-dasharray="{fraction * arc_len:.1f} 999"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{color}"/>'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
        '<g transform="translate(16,50)">'
        '<path d="M6 12 L16 -1 L52 -1 L62 12 Z" fill="#9aa0a6"/>'
        '<rect x="0" y="12" width="68" height="14" rx="7" fill="#9aa0a6"/>'
        '<circle cx="14" cy="28" r="6.5" fill="#3d3d3d"/>'
        '<circle cx="54" cy="28" r="6.5" fill="#3d3d3d"/>'
        '</g>'
        '</svg>'
    )


def render_image_choice(options, visuals, key, default):
    """A row of mutually-exclusive buttons, each with an icon above it -- behaves like a
    radio group, styled to show the selected option (type="primary") vs the rest."""
    if key not in st.session_state:
        st.session_state[key] = default
    cols = st.columns(len(options))
    for col, option in zip(cols, options):
        v = visuals[option]
        with col:
            st.markdown(
                f'<div style="text-align:center">{_car_gauge_icon_svg(v["gauge_fraction"], v["gauge_color"])}</div>',
                unsafe_allow_html=True,
            )
            selected = st.session_state[key] == option
            if st.button(
                option,
                key=f"{key}__{option}",
                width="stretch",
                type="primary" if selected else "secondary",
            ):
                st.session_state[key] = option
                st.rerun()
    return st.session_state[key]

st.set_page_config(page_title="Fleet maintenance monitoring", layout="wide")
st.title("Fleet maintenance monitoring")
st.caption(
    "Vehicle Fault Diagnosis & Repair Copilot -- XGBoost classifier, F2 = 0.594 on the held-out test set. "
    "Team NodePair."
)

st.subheader("Vehicle / usage profile")
vehicle = render_image_choice(
    options=list(VEHICLE_PROFILES.keys()),
    visuals=VEHICLE_PROFILES,
    key="vehicle_profile",
    default="Heavy user",
)

st.subheader("Scenario (pre-loaded 48h window)")
scenario_name = render_image_choice(
    options=list(SCENARIOS.keys()),
    visuals=SCENARIOS,
    key="scenario_name",
    default="Pre-fault escalation",
)

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
