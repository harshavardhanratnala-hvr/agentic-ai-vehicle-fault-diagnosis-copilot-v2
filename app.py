"""Fleet maintenance monitoring dashboard -- phase 3 of the capstone.

Calls src/classifier.py directly; this file is UI only, no model logic lives here.
Run with: streamlit run app.py
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
from classifier import (  # noqa: E402
    classify_fault,
    classify_fault_batch,
    classify_fault_baseline,
    classify_fault_batch_baseline,
    explain_prediction,
    FLAG_THRESHOLD,
)

# Internal keys stay plain; MODEL_LABELS is the only place the user-facing wording lives.
MODEL_LABELS = {"baseline": "No preloaded readings", "trained": "Trained model"}
MODEL_COLORS = {"baseline": "#eb6834", "trained": "#2a78d6"}  # validated categorical pair, fixed order
# Held-out test set numbers from notebook 12's scoreboard -- not re-derived here, just displayed.
MODEL_TEST_METRICS = {
    "baseline": {"recall": 0.516556, "precision": 0.132987, "F2": 0.327587},
    "trained": {"recall": 0.853989, "precision": 0.268013, "F2": 0.594172},
}

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


CALENDAR_FEATURE_LABELS = {
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "month_of_year": "Month of year",
    "hours_since_start": "Vehicle age (hours in service)",
}


def friendly_feature_name(feature: str) -> str:
    """Turn an engineered column name like 'Motor_Temp_roll_mean_24h' into something a
    fleet manager can read without knowing what a rolling window is."""
    if feature in CALENDAR_FEATURE_LABELS:
        return CALENDAR_FEATURE_LABELS[feature]
    for pattern, template in [
        (r"^(.*)_roll_mean_(\d+)h$", "{sensor} ({window}h average)"),
        (r"^(.*)_roll_std_(\d+)h$", "{sensor} ({window}h volatility)"),
        (r"^(.*)_lag_(\d+)h$", "{sensor} ({window}h ago)"),
        (r"^(.*)_delta_6h$", "{sensor} (change over last 6h)"),
    ]:
        m = re.match(pattern, feature)
        if m:
            groups = m.groups()
            sensor = groups[0].replace("_", " ")
            window = groups[1] if len(groups) > 1 else None
            return template.format(sensor=sensor, window=window)
    return feature.replace("_", " ")


def render_contribution_chart(factors, theme_type) -> str:
    """A diverging bar per factor: bars grow right (red) for impact that pushed risk up,
    left (blue) for impact that pushed it down -- same up/down color pairing used
    elsewhere in the dashboard, validated for both light and dark surfaces."""
    if theme_type == "dark":
        up_color, down_color, track_color, text_color, muted_color = (
            "#e66767", "#3987e5", "#2c2c2a", "#ffffff", "#c3c2b7",
        )
    else:
        # Brand neutrals (slate blue / sky blue) instead of generic gray, to match the
        # cream-and-blue theme -- the up/down semantic colors stay red/blue regardless.
        up_color, down_color, track_color, text_color, muted_color = (
            "#e34948", "#2a78d6", "#E8F1FA", "#3A5A78", "#6B87A0",
        )

    max_impact = max(abs(f["impact"]) for f in factors) or 1.0
    rows = []
    for f in factors:
        pct = abs(f["impact"]) / max_impact * 50
        is_up = f["impact"] > 0
        color = up_color if is_up else down_color
        anchor = "left" if is_up else "right"
        friendly = friendly_feature_name(f["feature"])
        direction_word = "up" if is_up else "down"
        tooltip = f"{friendly}: pushed risk {direction_word} by {abs(f['impact']):.3f}"
        # Every row must be a single line -- a blank line inside a raw HTML block passed to
        # st.markdown terminates that block early (CommonMark rule), silently dropping
        # everything after it; a multi-line f-string here reproduces that bug.
        row = (
            f'<div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid {track_color};">'
            f'<div style="flex:0 0 32%; text-align:right; color:{text_color}; font-size:0.9rem; line-height:1.3;">'
            f'{friendly}<br><span style="color:{muted_color}; font-size:0.72rem;">{f["feature"]} = {f["value"]:g}</span></div>'
            f'<div style="flex:1 1 auto; position:relative; height:18px; background:{track_color}55; border-radius:3px;" title="{tooltip}">'
            f'<div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:{muted_color};"></div>'
            f'<div style="position:absolute; {anchor}:50%; width:{pct:.1f}%; top:0; bottom:0; background:{color}; border-radius:3px;"></div>'
            f'</div>'
            f'<div style="flex:0 0 70px; color:{color}; font-weight:600; font-size:0.85rem;">{f["impact"]:+.3f}</div>'
            f'</div>'
        )
        rows.append(row)
    return f'<div>{"".join(rows)}</div>'


st.set_page_config(page_title="Road Safety Prediction Copilot", layout="wide")
theme_type = st.context.theme.type or "light"
st.markdown(
    '<div style="text-transform:uppercase; letter-spacing:0.14em; font-size:0.8rem; '
    'font-weight:600; color:#3A5A78;">NodePair</div>',
    unsafe_allow_html=True,
)
st.title("Road Safety Prediction Copilot")
st.caption(
    "Vehicle Fault Diagnosis & Repair Copilot -- XGBoost classifier, F2 = 0.594 on the held-out test set."
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

st.caption(SCENARIOS[scenario_name]["blurb"])
if vehicle != "Heavy user":
    st.info(
        "These three scenarios were pulled from real heavy_user data. Showing that vehicle's "
        "readings regardless of the profile selected above until scenarios exist for the others."
    )

st.subheader(f"Compare: {MODEL_LABELS['baseline']} vs. {MODEL_LABELS['trained']}")
st.caption(
    f"\"{MODEL_LABELS['baseline']}\" uses only the 10 raw sensor values for this moment, with no "
    f"historical trend. \"{MODEL_LABELS['trained']}\" adds 53 engineered signals -- rolling "
    "trends, recent history, vehicle age -- and is tuned to catch more real faults. Both numbers "
    "below are fixed, held-out test-set results, not specific to the scenario above."
)
metrics_df = pd.DataFrame({
    "Recall": [f"{MODEL_TEST_METRICS[k]['recall']:.0%}" for k in ["baseline", "trained"]],
    "Precision": [f"{MODEL_TEST_METRICS[k]['precision']:.0%}" for k in ["baseline", "trained"]],
    "F2 score": [f"{MODEL_TEST_METRICS[k]['F2']:.3f}" for k in ["baseline", "trained"]],
}, index=[MODEL_LABELS["baseline"], MODEL_LABELS["trained"]])
st.dataframe(metrics_df, width="stretch")

col_base, col_sep, col_trained = st.columns([10, 1, 10])
with col_base:
    run_baseline = st.button(
        f"Run prediction -- {MODEL_LABELS['baseline']}", key="run_baseline",
        width="stretch", icon=":material/query_stats:",
    )
with col_sep:
    sep_color = "#2c2c2a" if theme_type == "dark" else "#E8F1FA"
    st.markdown(
        f'<div style="border-left:1px solid {sep_color}; height:38px; margin:6px auto;"></div>',
        unsafe_allow_html=True,
    )
with col_trained:
    run_trained = st.button(
        f"Run prediction -- {MODEL_LABELS['trained']}", key="run_trained",
        type="primary", width="stretch", icon=":material/troubleshoot:",
    )

scenario_path = SCENARIOS_DIR / SCENARIOS[scenario_name]["file"]
readings = pd.read_csv(scenario_path, parse_dates=["timestamp"])

if run_baseline:
    st.session_state["baseline_result"] = classify_fault_baseline(readings)
    st.session_state["baseline_trend"] = classify_fault_batch_baseline(readings)
    st.session_state["baseline_scenario"] = scenario_name

if run_trained:
    st.session_state["trained_result"] = classify_fault(readings, vehicle_start_time=VEHICLE_START_TIME)
    st.session_state["trained_trend"] = classify_fault_batch(readings, vehicle_start_time=VEHICLE_START_TIME)
    st.session_state["trained_factors"] = explain_prediction(readings, vehicle_start_time=VEHICLE_START_TIME)
    st.session_state["trained_scenario"] = scenario_name

baseline_ready = st.session_state.get("baseline_scenario") == scenario_name
trained_ready = st.session_state.get("trained_scenario") == scenario_name

if baseline_ready or trained_ready:
    result_cols = st.columns(2)
    with result_cols[0]:
        st.markdown(f"**{MODEL_LABELS['baseline']}**")
        if baseline_ready:
            r = st.session_state["baseline_result"]
            st.metric("Risk status", r["risk_level"])
            st.metric("Probability of fault (next 6h)", f"{r['probability']:.0%}")
        else:
            st.caption("Click \"Run prediction\" above to see this model's result for the current scenario.")
    with result_cols[1]:
        st.markdown(f"**{MODEL_LABELS['trained']}**")
        if trained_ready:
            r = st.session_state["trained_result"]
            st.metric("Risk status", r["risk_level"])
            st.metric("Probability of fault (next 6h)", f"{r['probability']:.0%}")
        else:
            st.caption("Click \"Run prediction\" above to see this model's result for the current scenario.")
    st.caption(f"Decision threshold for both models: {FLAG_THRESHOLD:.0%}")

    st.subheader("Probability over the window")
    chart_series = {}
    if baseline_ready:
        chart_series[MODEL_LABELS["baseline"]] = (
            st.session_state["baseline_trend"].set_index("timestamp")["probability"]
        )
    if trained_ready:
        chart_series[MODEL_LABELS["trained"]] = (
            st.session_state["trained_trend"].set_index("timestamp")["probability"]
        )
    chart_colors = [MODEL_COLORS[k] for k in ["baseline", "trained"] if
                     (k == "baseline" and baseline_ready) or (k == "trained" and trained_ready)]
    st.line_chart(pd.DataFrame(chart_series), color=chart_colors)

    if trained_ready:
        st.subheader(f"Why: top contributing readings ({MODEL_LABELS['trained']})")
        st.caption(
            "From the model's own feature contributions (XGBoost's built-in SHAP values) -- not RAG, "
            "no documents involved."
        )
        factors = st.session_state["trained_factors"]
        st.markdown(
            f"Out of the {st.session_state['trained_result']['n_features_used']} signals the model "
            "looks at, these are the ones that mattered most for **this specific prediction**. Each "
            "bar shows how much that one reading pushed the model's risk score up (toward \"fault "
            "likely\") or down (toward \"normal\") for this window -- the longer the bar, the bigger "
            "that reading's influence on the result you're looking at right now, not on the model in general."
        )
        st.markdown(render_contribution_chart(factors, theme_type), unsafe_allow_html=True)

    with st.expander("Raw readings for this window"):
        st.dataframe(readings, width="stretch")

st.divider()
st.caption(
    "Model: models/xgboost_all_vehicles.joblib, 53 selected features, trained on the full "
    "140,067-row training set (no subsampling). See docs/Business_Goal_and_Process.md and "
    "docs/Classifier_and_Dashboard_FAQ.md for the full writeup."
)
st.caption("© NodePair")
