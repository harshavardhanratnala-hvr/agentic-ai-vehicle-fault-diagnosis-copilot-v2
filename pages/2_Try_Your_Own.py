"""Try Your Own page: type in one current reading and check it against the baseline model.

Corresponds to step 4 of the dashboard flow. Baseline-only, deliberately: this page's whole
premise is "one reading, no history," which matches exactly what the baseline model is (present
moment only). The advanced model needs a real 24h window, which doesn't fit this page's mental
model even though it's technically possible to keep 23 real hours and swap in 1 manual one --
that combination just confused more than it clarified. Baseline vs. advanced comparisons happen
on Live Diagnosis (a full real scenario) and Compare Models instead.
"""

import time

import pandas as pd
import streamlit as st

from dashboard_lib import (
    RAW_SENSOR_COLS,
    SENSOR_RANGES,
    classify_fault_baseline,
    page_header,
    render_result_card,
    render_sidebar,
)

vehicle, scenario_name, readings_full = render_sidebar()

page_header("Try Your Own", "Check a single reading against the baseline model.")

st.caption("No history needed here -- that's the point of the baseline model. For the advanced model, which needs a real 24h window, use Live Diagnosis instead.")

latest_row = readings_full.iloc[-1]
manual_values = {}

with st.container(key="card_manual_inputs"):
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

st.write("")
predict_manual = st.button("Check this reading", key="predict_manual", width="stretch", type="primary")
if predict_manual:
    with st.spinner("Checking the baseline model against your reading..."):
        time.sleep(0.3)
        manual_row = latest_row.copy()
        for col, val in manual_values.items():
            manual_row[col] = val
        manual_df = pd.DataFrame([manual_row])

        st.session_state["baseline_result"] = classify_fault_baseline(manual_df)
        st.session_state["baseline_loaded_at"] = pd.Timestamp.now().strftime("%H:%M:%S") + " (manual reading)"

has_baseline = "baseline_result" in st.session_state

if has_baseline:
    st.markdown('<div class="section-label" style="margin-top:1.6rem;">Result</div>', unsafe_allow_html=True)
    render_result_card("Baseline result", st.session_state["baseline_result"])
    st.write("")
    st.page_link("pages/1_Live_Diagnosis.py", label="See the advanced model on a full scenario")
else:
    st.info("Set your reading above and check it to see a result.")
