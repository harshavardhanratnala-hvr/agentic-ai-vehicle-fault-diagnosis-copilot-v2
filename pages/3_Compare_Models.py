"""Compare Models page: baseline vs. advanced, side by side, on whatever was last loaded.

Corresponds to step 3 of the dashboard flow. Reads from session_state populated by either
Live Diagnosis or Try Your Own -- it doesn't load anything itself, since the whole point of
this page is showing two results the visitor already produced elsewhere.
"""

import streamlit as st

from dashboard_lib import _radial_gauge_svg, page_header, render_sidebar, risk_colors

render_sidebar()

page_header("Compare Models", "Same scenario, two models, side by side.")

has_baseline = "baseline_result" in st.session_state
has_advanced = "advanced_result" in st.session_state

if not (has_baseline and has_advanced):
    st.info("Load both models first, on Live Diagnosis or Try Your Own.")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/1_Live_Diagnosis.py", label="Go to Live Diagnosis")
    with c2:
        st.page_link("pages/2_Try_Your_Own.py", label="Go to Try Your Own")
else:
    b = st.session_state["baseline_result"]
    a = st.session_state["advanced_result"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown(_radial_gauge_svg(b["probability"], risk_colors(b["risk_level"])[0], "BASELINE", size=170), unsafe_allow_html=True)
        st.markdown(
            f'<p style="margin-top:0.6rem; color:#64748b;">Present readings only &middot; F2 0.328</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown(_radial_gauge_svg(a["probability"], risk_colors(a["risk_level"])[0], "ADVANCED", size=170), unsafe_allow_html=True)
        st.markdown(
            f'<p style="margin-top:0.6rem; color:#64748b;">24h rolling/lag history &middot; F2 0.706</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.caption("The gap above is what 24h of history buys the advanced model over the baseline.")
