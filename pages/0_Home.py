"""Home page: what this dashboard does, and a way into the flow.

Content-equivalent to the old single-page app.py intro, trimmed down: one short line per
idea instead of full paragraphs, no emoji icons.
"""

import streamlit as st

from dashboard_lib import _car_gauge_icon_svg, render_sidebar

render_sidebar()

# Same car+speedometer icon already used next to each vehicle option in the sidebar, just
# larger and in the light "on_dark" variant so it reads clearly against the navy hero --
# reusing an asset already in the app instead of pulling in new stock imagery.
hero_icon = _car_gauge_icon_svg(0.92, "#f97316", size=150, on_dark=True)

st.markdown(
    f"""
    <div class="hero" style="display:flex; align-items:center; justify-content:space-between; gap:1.5rem;">
        <div>
            <h1>Fault Early Warning Dashboard</h1>
            <p class="sub">Predicts battery and drivetrain faults up to 6 hours ahead, from live telemetry.</p>
            <div class="chip-row">
                <span class="chip">53 engineered features</span>
                <span class="chip">Team NodePair</span>
            </div>
        </div>
        <div style="flex-shrink:0; opacity:0.9;">{hero_icon}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
# One accent color per step, reusing the palette already used elsewhere (blue = baseline/
# primary, purple = advanced, teal/amber new but restrained -- a small color dot, not a
# full icon, so this stays a simple accent rather than another emoji-style decoration.
steps = [
    ("Live Diagnosis", "Load a scenario, see the risk call and why.", "#2563eb"),
    ("Manual Sensor Entry (MSE)", "Edit a reading, watch both models react.", "#7c3aed"),
    ("Compare Models", "Advanced Model I vs. Advanced Model II, side by side.", "#0891b2"),
    ("About", "The data, the models, the metric.", "#d97706"),
]
for col, (title, desc, accent) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(
            f"""
            <div class="card step-card">
                <div style="width:14px; height:14px; border-radius:4px; background:{accent}; margin-bottom:0.8rem;"></div>
                <h4>{title}</h4>
                <p class="desc">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.page_link("pages/1_Live_Diagnosis.py", label="Start with Live Diagnosis")
