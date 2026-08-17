"""Live Diagnosis page: load both models on the selected vehicle/scenario and see the result.

Corresponds to steps 1 and 2 of the dashboard flow (select -> quick check -> deep look).
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_lib import (
    FLAG_THRESHOLD,
    PRIMARY,
    DANGER,
    classify_fault,
    classify_fault_batch,
    explain_prediction,
    classify_fault_baseline,
    page_header,
    render_result_card,
    render_sidebar,
    VEHICLE_START_TIME,
)

vehicle, scenario_name, readings_full = render_sidebar()

page_header("Live Diagnosis", "Load the scenario into both models and see the result.")

col_baseline, col_advanced = st.columns(2)
with col_baseline:
    st.markdown(
        """
        <div class="card">
            <span class="badge badge-blue">BASELINE</span>
            <span class="badge" style="background:#f1f5f9; color:#334155;">F2 0.328</span>
            <h4>Present reading only</h4>
            <p class="desc">Logistic regression, no history needed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    load_present = st.button("Load present reading", key="load_baseline", width="stretch")
    if st.session_state.get("baseline_loaded_at"):
        st.caption(f"Loaded at {st.session_state['baseline_loaded_at']}")

with col_advanced:
    st.markdown(
        """
        <div class="card">
            <span class="badge badge-purple">ADVANCED</span>
            <span class="badge" style="background:#f1f5f9; color:#334155;">F2 0.706</span>
            <h4>Last 24 hours</h4>
            <p class="desc">Tuned XGBoost, 53 engineered features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    load_history = st.button("Load last 24 hours", key="load_advanced", width="stretch", type="primary")
    if st.session_state.get("advanced_loaded_at"):
        st.caption(f"Loaded at {st.session_state['advanced_loaded_at']}")

if load_present:
    with st.spinner("Reading current sensor values..."):
        time.sleep(0.5)
        st.session_state["baseline_result"] = classify_fault_baseline(readings_full)
        st.session_state["baseline_loaded_at"] = pd.Timestamp.now().strftime("%H:%M:%S")

if load_history:
    with st.spinner("Pulling the last 24 hours of telemetry and computing rolling and lag features..."):
        time.sleep(0.7)
        st.session_state["advanced_result"] = classify_fault(readings_full, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_trend"] = classify_fault_batch(readings_full, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_factors"] = explain_prediction(readings_full, vehicle_start_time=VEHICLE_START_TIME)
        st.session_state["advanced_loaded_at"] = pd.Timestamp.now().strftime("%H:%M:%S")

has_baseline = "baseline_result" in st.session_state
has_advanced = "advanced_result" in st.session_state

if not (has_baseline or has_advanced):
    st.info("Load either model above to see a result.")

if has_baseline or has_advanced:
    st.markdown('<div class="section-label" style="margin-top:1.6rem;">Results</div>', unsafe_allow_html=True)

if has_baseline:
    render_result_card("Baseline result", st.session_state["baseline_result"])
    st.write("")

if has_advanced:
    result = st.session_state["advanced_result"]
    trend = st.session_state["advanced_trend"]
    factors = st.session_state["advanced_factors"]

    render_result_card("Advanced result", result)
    st.write("")

    left, right = st.columns([1.4, 1])
    with left, st.container(key="card_trend_chart"):
        st.markdown("#### Probability over the window")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["timestamp"], y=trend["probability"], mode="lines", fill="tozeroy",
            line=dict(color=PRIMARY, width=2.5), fillcolor="rgba(37,99,235,0.12)",
            hovertemplate="%{x|%b %d %H:%M}<br>Probability: %{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=FLAG_THRESHOLD, line_dash="dash", line_color=DANGER,
                       annotation_text="Flag threshold", annotation_position="top left")
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(tickformat=".0%", gridcolor="#eef2f7", range=[0, 1]),
            xaxis=dict(gridcolor="#eef2f7"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right, st.container(key="card_why_factors"):
        st.markdown("#### Why")
        st.caption("Top contributing readings.")
        max_abs = max(abs(f["impact"]) for f in factors) or 1.0
        for f in factors:
            up = f["impact"] > 0
            bar_color = DANGER if up else "#16a34a"
            width_pct = abs(f["impact"]) / max_abs * 100
            st.markdown(
                f"""
                <div class="factor-row">
                    <div style="flex:1;">
                        <div class="factor-name">{f['feature']}</div>
                        <div style="background:#eef2f7; border-radius:6px; height:6px; margin-top:5px; width:100%;">
                            <div style="background:{bar_color}; border-radius:6px; height:6px; width:{width_pct:.0f}%;"></div>
                        </div>
                    </div>
                    <div class="factor-val" style="text-align:right; margin-left:0.7rem;">
                        {f['value']:g}<br>{f['impact']:+.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Raw readings for this window"):
        st.dataframe(readings_full, width="stretch")

if has_baseline or has_advanced:
    st.write("")
    n1, n2 = st.columns(2)
    with n1:
        st.page_link("pages/2_Try_Your_Own.py", label="Try editing a reading yourself")
    with n2:
        if has_baseline and has_advanced:
            st.page_link("pages/3_Compare_Models.py", label="Compare baseline vs. advanced")
