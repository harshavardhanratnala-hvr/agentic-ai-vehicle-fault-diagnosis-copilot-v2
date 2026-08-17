"""Home page: what this dashboard does, and a way into the flow.

Content-equivalent to the old single-page app.py intro, trimmed down: one short line per
idea instead of full paragraphs, no emoji icons.
"""

import streamlit as st

from dashboard_lib import render_sidebar

render_sidebar()

st.markdown(
    """
    <div class="hero">
        <h1>Fault Early Warning Dashboard</h1>
        <p class="sub">Predicts battery and drivetrain faults up to 6 hours ahead, from live telemetry.</p>
        <div class="chip-row">
            <span class="chip">XGBoost · F2 0.706</span>
            <span class="chip">53 engineered features</span>
            <span class="chip">Team NodePair</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
steps = [
    ("Live Diagnosis", "Load a scenario, see the risk call and why."),
    ("Try Your Own", "Edit a reading, watch both models react."),
    ("Compare Models", "Baseline vs. advanced, side by side."),
    ("About", "The data, the models, the metric."),
]
for col, (title, desc) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <h4>{title}</h4>
                <p class="desc">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.page_link("pages/1_Live_Diagnosis.py", label="Start with Live Diagnosis")
