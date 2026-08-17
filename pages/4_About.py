"""About page: the data, the models, the metric, in brief."""

import streamlit as st

from dashboard_lib import page_header, render_sidebar_minimal

render_sidebar_minimal()

page_header("About", "The data, the models, the metric.")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="card">
            <span class="badge badge-blue">BASELINE</span>
            <h4>Logistic Regression</h4>
            <p class="desc">Present moment only, no history. F2 = 0.328.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
            <span class="badge badge-purple">ADVANCED</span>
            <h4>XGBoost, tuned</h4>
            <p class="desc">53 engineered features, 24h history. F2 = 0.706.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown('<div class="section-label">Why F2, not accuracy</div>', unsafe_allow_html=True)
st.write("Missing a real fault costs more than a false alarm, so recall matters twice as much as precision here.")

st.write("")
st.markdown('<div class="section-label">The data</div>', unsafe_allow_html=True)
st.write("175,176 hourly readings across 4 vehicles, 5 years. Kaggle EV Sensors dataset, CC BY 4.0.")

st.write("")
st.markdown('<div class="section-label">A known limitation</div>', unsafe_allow_html=True)
st.write("Only 4 usage profiles. More vehicles would help confirm the pattern generalizes.")
