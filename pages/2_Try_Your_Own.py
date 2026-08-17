"""Try Your Own page: edit the most recent hour of sensor readings and see both models react.

Corresponds to step 4 of the dashboard flow. This used to be a collapsed expander at the
bottom of the single-page app; it's now its own page so it gets equal billing in the nav
instead of being easy to miss.
"""

import time

import pandas as pd
import streamlit as st

from dashboard_lib import (
    RAW_SENSOR_COLS,
    SENSOR_RANGES,
    VEHICLE_START_TIME,
    classify_fault,
    classify_fault_batch,
    classify_fault_baseline,
    explain_prediction,
    page_header,
    render_result_card,
    render_sidebar,
)

vehicle, scenario_name, readings_full = render_sidebar()

page_header("Try Your Own", "Set your own current reading and watch both models react.")

latest_row = readings_full.iloc[-1]
manual_values = {}

st.markdown('<div class="card">', unsafe_allow_html=True)
cols = st.columns(2)
for i, col in enumerate(RAW_SENSOR_COLS):
    with cols[i % 2]:
        if col == "Charging_Voltage":
            options = [240, 400]
            default = int(latest_row[col]) if int(latest_row[col]) in options else 240
            manual_values[col] = st.selectbox(
                "Charging Voltage (V)", options, index=options.index(default), key="manual_Charging_Voltage"
            )
        else:
            lo, hi, step, unit = SENSOR_RANGES[col]
            label = f"{col.replace('_', ' ')} ({unit})" if unit else col.replace("_", " ")
            manual_values[col] = st.slider(
                label, min_value=float(lo), max_value=float(hi),
                value=float(min(max(latest_row[col], lo), hi)), step=float(step),
                key=f"manual_{col}",
            )
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
predict_manual = st.button("Apply this reading to both models", key="predict_manual", width="stretch", type="primary")
if predict_manual:
    with st.spinner("Recomputing both models around your edited reading..."):
        time.sleep(0.4)
        manual_window = readings_full.copy()
        for col, val in manual_values.items():
            manual_window.loc[manual_window.index[-1], col] = val

        stamp = pd.Timestamp.now().strftime("%H:%M:%S") + " (manual reading)"
        st.session_state["baseline_result"] = classify_fault_baseline(manual_window)
        st.session_state["baseline_loaded_at"] = stamp
        st.session_state["advanced_result"] = classify_fault(manual_window, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_trend"] = classify_fault_batch(manual_window, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_factors"] = explain_prediction(manual_window, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_loaded_at"] = stamp

has_baseline = "baseline_result" in st.session_state
has_advanced = "advanced_result" in st.session_state

if has_baseline or has_advanced:
    st.markdown('<div class="section-label" style="margin-top:1.6rem;">Results</div>', unsafe_allow_html=True)
    if has_baseline:
        render_result_card("Baseline result", st.session_state["baseline_result"])
        st.write("")
    if has_advanced:
        render_result_card("Advanced result", st.session_state["advanced_result"])
    st.write("")
    st.page_link("pages/1_Live_Diagnosis.py", label="Back to Live Diagnosis for the full trend")
else:
    st.info("Set your readings above and apply them to see a result.")
