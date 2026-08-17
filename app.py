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

pg = st.navigation(
    [
        st.Page("pages/0_Home.py", title="Home", default=True),
        st.Page("pages/1_Live_Diagnosis.py", title="Live Diagnosis"),
        st.Page("pages/2_Try_Your_Own.py", title="Try Your Own"),
        st.Page("pages/3_Compare_Models.py", title="Compare Models"),
        st.Page("pages/4_About.py", title="About"),
    ]
)
pg.run()
