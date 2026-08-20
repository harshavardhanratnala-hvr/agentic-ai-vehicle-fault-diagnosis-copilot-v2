"""Entry point / router for the multi-page dashboard.

Uses st.navigation so the sidebar nav shows plain, clean labels (Home, Live Diagnosis, ...)
instead of Streamlit's filename-derived defaults. Run with: streamlit run app.py
"""

import streamlit as st

from dashboard_lib import inject_theme

st.set_page_config(
    page_title="Fault Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

# Grouped nav (section headers cost nothing and make the five pages easier to scan at a
# glance). Dropped the Material icons here -- three CSS attempts at making them visible
# against the dark sidebar didn't hold up, and the plain text version reads cleanly without
# them, so it's not worth further guessing without a browser to actually inspect.
pg = st.navigation(
    {
        "Overview": [
            st.Page("pages/0_Home.py", title="Home", default=True),
        ],
        "Diagnostics": [
            st.Page("pages/1_Live_Diagnosis.py", title="Live Diagnosis"),
            st.Page("pages/2_Try_Your_Own.py", title="Manual Sensor Entry (MSE)"),
            st.Page("pages/3_Compare_Models.py", title="Compare Models"),
        ],
        "Documentation": [
            st.Page("pages/4_About.py", title="About"),
        ],
    }
)
pg.run()