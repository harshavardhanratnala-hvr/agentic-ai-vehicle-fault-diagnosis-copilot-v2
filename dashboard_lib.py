"""Shared UI plumbing for the multi-page Streamlit dashboard.

Every page under pages/ (and app.py, the Overview/home page) imports from here: theme CSS,
color constants, scenario/vehicle data, the SVG gauge helpers, the sidebar picker, and the
classifier functions re-exported for convenience. No model logic lives here, same rule as
before -- this file is UI plumbing only, src/classifier.py still owns the actual predictions.

Why a shared module instead of copy-pasting into every page: the sidebar picker (vehicle +
scenario) has to stay in sync across pages, and Streamlit keeps session_state shared across
pages in the same session automatically -- so one set of session_state keys, written by one
function, is what makes "pick a scenario on Live Diagnosis, see it reflected on Compare
Models" work at all.
"""

import math
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
from classifier import (  # noqa: E402
    classify_fault,
    classify_fault_batch,
    classify_fault_baseline,
    explain_prediction,
    FLAG_THRESHOLD,
    RAW_SENSOR_COLS,
)

# ---------------------------------------------------------------------------
# Constants (unchanged from the original single-page app.py)
# ---------------------------------------------------------------------------

SENSOR_RANGES = {
    "SOC": (0.0, 100.0, 0.5, "%"),
    "SOH": (70.0, 100.0, 0.1, "%"),
    "Charging_Cycles": (0, 700, 1, ""),
    "Battery_Temp": (10.0, 75.0, 0.5, "°C"),
    "Motor_RPM": (0.0, 5000.0, 10.0, "rpm"),
    "Motor_Torque": (0.0, 1000.0, 5.0, "Nm"),
    "Motor_Temp": (50.0, 120.0, 0.5, "°C"),
    "Brake_Pad_Wear": (0.0, 100.0, 0.5, "%"),
    "Tire_Pressure": (20.0, 38.0, 0.1, "psi"),
}

SCENARIOS_DIR = BASE_DIR / "data" / "processed" / "scenarios"
VEHICLE_START_TIME = "2020-01-01 00:00:00"

INK = "#0f172a"
MUTED = "#64748b"
CARD_BG = "#ffffff"
BORDER = "#e2e8f0"
PRIMARY = "#2563eb"
PRIMARY_DARK = "#1d4ed8"
SUCCESS = "#16a34a"
WARNING = "#d97706"
DANGER = "#dc2626"

SCENARIOS = {
    "Normal driving": {
        "key": "normal",
        "blurb": "A calm 48h window, no fault episode nearby.",
        "gauge_fraction": 0.15,
        "gauge_color": SUCCESS,
    },
    "Gradual degradation": {
        "key": "escalating",
        "blurb": "48h window with risk climbing toward a fault, not yet flagged.",
        "gauge_fraction": 0.55,
        "gauge_color": WARNING,
    },
    "Pre-fault escalation": {
        "key": "prefault",
        "blurb": "48h window ending right at a real fault episode in the data.",
        "gauge_fraction": 0.92,
        "gauge_color": DANGER,
    },
}

VEHICLE_PROFILES = {
    "Rare user": {"key": "rare_user", "gauge_fraction": 0.15, "gauge_color": SUCCESS},
    "Moderate user": {"key": "moderate_user", "gauge_fraction": 0.45, "gauge_color": "#ca8a04"},
    "Daily user": {"key": "daily_user", "gauge_fraction": 0.7, "gauge_color": "#ea580c"},
    "Heavy user": {"key": "heavy_user", "gauge_fraction": 0.92, "gauge_color": DANGER},
}

# ---------------------------------------------------------------------------
# Theme (identical CSS to the original app.py, factored out so every page gets it)
# ---------------------------------------------------------------------------

def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; font-size: 1.15rem; }
        .stApp { background: #f4f6fb; }

        section[data-testid="stSidebar"] { background: #0f172a; }
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown {
            color: #e5e9f2 !important;
        }
        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
            color: #a3adc2 !important; font-size: 0.95rem !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            font-size: 1.1rem !important; padding: 0.6rem 1rem !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: #1e293b !important; color: #e5e9f2 !important;
            border: 1px solid #334155 !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: #263449 !important; border-color: #475569 !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #2563eb !important; color: #ffffff !important; border: none !important;
        }

        /* Sidebar page-nav: match the dark sidebar theme instead of Streamlit's default */
        section[data-testid="stSidebarNav"] { background: #0f172a; padding-top: 0.5rem; }
        section[data-testid="stSidebarNav"] a { color: #e5e9f2 !important; border-radius: 8px; }
        section[data-testid="stSidebarNav"] a:hover { background: #1e293b !important; }
        section[data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #2563eb !important; color: #ffffff !important;
        }

        div.block-container { padding-top: 2.4rem; max-width: 1200px; }

        .hero {
            background: linear-gradient(120deg, #0f172a 0%, #1e2a5e 55%, #2545a8 100%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px -12px rgba(15,23,42,0.45);
        }
        .hero h1 { margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; }
        .hero p.sub { margin: 0.5rem 0 1.1rem 0; color: #c7d2fe; font-size: 1.15rem; }
        .chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .chip {
            display: inline-flex; align-items: center; gap: 0.35rem;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
            color: #eef2ff; padding: 0.35rem 0.85rem; border-radius: 999px;
            font-size: 0.9rem; font-weight: 600;
        }

        .card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
            padding: 1.4rem 1.5rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            height: 100%;
        }
        .card h4 { margin: 0 0 0.5rem 0; font-size: 1.35rem; color: #0f172a; font-weight: 700; }
        .card p.desc { color: #64748b; font-size: 1.02rem; line-height: 1.5; margin-bottom: 0.9rem; }

        /* Home's 4 step cards: fixed min-height so a 2-line title ("Live Diagnosis" wrapping)
           doesn't make that card taller than its 1-line neighbors. */
        .step-card { min-height: 180px; display: flex; flex-direction: column; }
        /* Reserve 2 lines of title height on every step card, so a 1-line title ("About")
           and a 2-line title ("Live Diagnosis") still leave the description starting at the
           same y position instead of staggering across the row. */
        .step-card h4 { line-height: 1.25; min-height: 3.4rem; }

        /* st.container(key="card_...") -- used anywhere a "card" needs to wrap a native
           Streamlit widget (chart, slider, dataframe) that can't go inside a raw HTML
           string. Every such key is prefixed "card_" so this one rule styles all of them. */
        div[class*="st-key-card_"] {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
            padding: 1.4rem 1.5rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }

        .badge {
            display: inline-block; font-size: 0.82rem; font-weight: 700; padding: 0.2rem 0.6rem;
            border-radius: 6px; margin-bottom: 0.6rem; letter-spacing: 0.03em;
        }
        .badge-blue { background: #dbeafe; color: #1e40af; }
        .badge-purple { background: #ede9fe; color: #5b21b6; }

        .stButton > button {
            border-radius: 10px; font-weight: 600; border: none; font-size: 1.1rem;
            transition: transform 0.05s ease-in-out;
        }
        .stButton > button:active { transform: scale(0.98); }
        .stButton > button[kind="primary"] {
            background: #2563eb; box-shadow: 0 4px 10px -4px rgba(37,99,235,0.6);
        }
        .stButton > button[kind="primary"]:hover { background: #1d4ed8; }

        .section-label {
            font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; color: #2563eb;
            text-transform: uppercase; margin-bottom: 0.4rem;
        }

        .page-title { font-size: 2.3rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0 0.3rem 0; line-height: 1.25; }
        .page-sub { color: #64748b; font-size: 1.1rem; margin-bottom: 1.6rem; }

        .result-wrap { display: flex; align-items: center; gap: 1.6rem; }
        .result-copy h3 { margin: 0; font-size: 1.6rem; color: #0f172a; }
        .result-copy p { margin: 0.2rem 0 0 0; color: #64748b; font-size: 1.02rem; }
        .risk-pill {
            display: inline-flex; align-items: center; gap: 0.4rem; font-weight: 700;
            font-size: 0.92rem; padding: 0.35rem 0.9rem; border-radius: 999px; margin-top: 0.5rem;
        }
        .factor-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.6rem 0; border-bottom: 1px solid #e2e8f0; font-size: 0.98rem;
        }
        .factor-row:last-child { border-bottom: none; }
        .factor-name { font-weight: 600; color: #0f172a; }
        .factor-val { color: #64748b; font-size: 0.88rem; }

        div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid #e2e8f0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    """Small consistent page title used at the top of every non-Overview page, so it's clear
    which step of the flow you're on even though each step is now its own page."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable SVG components (unchanged)
# ---------------------------------------------------------------------------

def _car_gauge_icon_svg(fraction: float, color: str, size: int = 64, on_dark: bool = False) -> str:
    """The car+speedometer icon used next to each vehicle/scenario option in the sidebar.
    on_dark swaps the body/wheel fill to a light slate so the same icon reads clearly when
    placed on the navy hero background instead of the sidebar's white row background."""
    body_color = "#cbd5e1" if on_dark else "#334155"
    wheel_color = "#94a3b8" if on_dark else "#0f172a"
    cx, cy, r = 50, 40, 30
    angle = math.radians(180 - fraction * 180)
    nx, ny = cx + (r - 6) * math.cos(angle), cy - (r - 6) * math.sin(angle)
    arc_len = math.pi * r
    return (
        f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="{body_color}" '
        'stroke-width="7" stroke-linecap="round" opacity="0.35"/>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="{color}" '
        f'stroke-width="7" stroke-linecap="round" stroke-dasharray="{fraction * arc_len:.1f} 999"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{color}"/>'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
        '<g transform="translate(16,50)">'
        f'<path d="M6 12 L16 -1 L52 -1 L62 12 Z" fill="{body_color}"/>'
        f'<rect x="0" y="12" width="68" height="14" rx="7" fill="{body_color}"/>'
        f'<circle cx="14" cy="28" r="6.5" fill="{wheel_color}"/>'
        f'<circle cx="54" cy="28" r="6.5" fill="{wheel_color}"/>'
        '</g>'
        '</svg>'
    )


def _radial_gauge_svg(fraction: float, color: str, label: str, size: int = 168) -> str:
    r = 62
    circumference = 2 * math.pi * r
    filled = max(0.0, min(1.0, fraction)) * circumference
    cx = cy = 80
    return (
        f'<svg viewBox="0 0 160 160" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="14"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="14" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.1f} 999" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="30" font-weight="800" '
        f'fill="{INK}" font-family="Inter, sans-serif">{fraction*100:.0f}%</text>'
        + (
            f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="11" font-weight="600" '
            f'fill="{MUTED}" font-family="Inter, sans-serif">{label}</text>'
            if label else ""
        )
        + '</svg>'
    )


def render_image_choice(options, visuals, key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    for option in options:
        v = visuals[option]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div style="text-align:center">{_car_gauge_icon_svg(v["gauge_fraction"], v["gauge_color"])}</div>',
                unsafe_allow_html=True,
            )
        with c2:
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


def risk_colors(risk_level: str):
    return (SUCCESS, "#dcfce7", "#166534") if risk_level == "Normal" else (DANGER, "#fee2e2", "#991b1b")


def render_result_card(title: str, result: dict, note: str = None):
    color, pill_bg, pill_fg = risk_colors(result["risk_level"])
    # st.columns is a real widget, not raw HTML, so this has to be a real container --
    # opening a <div class="card"> in one st.markdown call and closing it in another
    # doesn't nest (each st.markdown call is its own isolated block), which was rendering
    # as an empty white box followed by unstyled content. st.container(key=...) sidesteps
    # that: see the "card_" CSS rule in inject_theme.
    card_key = "card_result_" + "".join(c if c.isalnum() else "_" for c in title.lower())
    with st.container(key=card_key):
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown(
                f'<div style="text-align:center">{_radial_gauge_svg(result["probability"], color, "PROBABILITY")}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="result-copy">
                    <h3>{title}</h3>
                    <span class="risk-pill" style="background:{pill_bg}; color:{pill_fg};">{result['risk_level'].upper()}</span>
                    <p>Threshold {FLAG_THRESHOLD:.0%} &middot; {result['timestamp']}</p>
                    {f'<p>{note}</p>' if note else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar -- shared across every page so the selected vehicle/scenario stays in sync
# ---------------------------------------------------------------------------

def render_sidebar():
    """Renders the vehicle/scenario picker + about-models expander in the sidebar, and returns
    (vehicle, scenario_name, readings_full) for the current selection. Call this once at the
    top of every page -- it's cheap, and it's what keeps the selection in sync across pages."""
    with st.sidebar:
        st.markdown("**Vehicle Usage Profile**")
        vehicle = render_image_choice(
            options=list(VEHICLE_PROFILES.keys()),
            visuals=VEHICLE_PROFILES,
            key="vehicle_profile",
            default="Heavy user",
        )
        st.markdown("---")
        st.markdown("**Scenario**")
        scenario_name = render_image_choice(
            options=list(SCENARIOS.keys()),
            visuals=SCENARIOS,
            key="scenario_name",
            default="Pre-fault escalation",
        )
        st.caption(SCENARIOS[scenario_name]["blurb"])
        st.markdown("---")
        with st.expander("About the models"):
            st.write("**Advanced Model I** — logistic regression, present readings only.")
            st.write("**Advanced Model II** — tuned XGBoost, 24h history.")

    # If the scenario or vehicle changed since the last load, drop stale results rather than
    # showing a prediction that no longer matches what's selected in the sidebar.
    _selection_key = (vehicle, scenario_name)
    if st.session_state.get("last_selection") != _selection_key:
        st.session_state["last_selection"] = _selection_key
        st.session_state.pop("baseline_result", None)
        st.session_state.pop("advanced_result", None)
        st.session_state.pop("baseline_loaded_at", None)
        st.session_state.pop("advanced_loaded_at", None)

    scenario_file = f'{VEHICLE_PROFILES[vehicle]["key"]}_{SCENARIOS[scenario_name]["key"]}.csv'
    scenario_path = SCENARIOS_DIR / scenario_file
    readings_full = pd.read_csv(scenario_path, parse_dates=["timestamp"])
    return vehicle, scenario_name, readings_full


def render_sidebar_minimal():
    """Lighter sidebar for pages that don't depend on the vehicle/scenario selection (About).
    No picker to keep in sync, so this skips it rather than showing an irrelevant filter."""
    with st.sidebar:
        with st.expander("About the models"):
            st.write("**Advanced Model I** — logistic regression, present readings only.")
            st.write("**Advanced Model II** — tuned XGBoost, 24h history.")
