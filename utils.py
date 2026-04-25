import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Design tokens ─────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "Good":         "#00C853",
    "Satisfactory": "#8BC34A",
    "Moderate":     "#FFD600",
    "Poor":         "#FF6D00",
    "Very Poor":    "#DD2C00",
    "Severe":       "#6A1A4C",
}
STATUS_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

SEASON_MAP = {
    1: "Winter",  2: "Winter",  3: "Pre-monsoon",
    4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Post-monsoon", 10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}
SEASON_COLORS = {
    "Winter":       "#90CAF9",
    "Pre-monsoon":  "#A5D6A7",
    "Monsoon":      "#80DEEA",
    "Post-monsoon": "#FFCC80",
}
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ── CSS (injected on every page via inject_css()) ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0d1b2a 0%, #1a2e45 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.main .block-container { padding-top: 1.8rem; max-width: 1150px; }

.page-hero {
    background: linear-gradient(130deg, #0369a1 0%, #0ea5e9 60%, #38bdf8 100%);
    border-radius: 14px; padding: 26px 30px; margin-bottom: 24px; color: white;
}
.page-hero h1 { font-size: 1.85rem; font-weight: 700; margin:0 0 5px; color:white; }
.page-hero p  { font-size: 0.95rem; margin:0; opacity:0.9; color:white; }

.kpi-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-label { font-size: 0.75rem; font-weight: 600; color: #64748b;
             text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
.kpi-sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }

.section-title {
    font-size: 1.05rem; font-weight: 600; color: #1e293b;
    padding-bottom: 8px; border-bottom: 2px solid #e2e8f0;
    margin: 28px 0 14px;
}

.insight-chip {
    display: inline-block; background: #f0f9ff; color: #0369a1;
    border: 1px solid #bae6fd; border-radius: 20px;
    padding: 4px 12px; font-size: 0.8rem; font-weight: 500; margin: 3px;
}
.warn-chip {
    display: inline-block; background: #fff7ed; color: #c2410c;
    border: 1px solid #fed7aa; border-radius: 20px;
    padding: 4px 12px; font-size: 0.8rem; font-weight: 500; margin: 3px;
}

.insight-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-left: 4px solid #0ea5e9; border-radius: 10px;
    padding: 14px 18px; font-size: 0.9rem; margin: 8px 0; color: #334155;
}
.warn-box {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 4px solid #f97316; border-radius: 10px;
    padding: 14px 18px; font-size: 0.9rem; margin: 8px 0; color: #7c2d12;
}
.good-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 10px;
    padding: 14px 18px; font-size: 0.9rem; margin: 8px 0; color: #14532d;
}

.nav-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #475569 !important; margin: 16px 0 4px;
}

[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
</style>
"""


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("aqi.csv")
    df.drop(columns=["note", "unit"], errors="ignore", inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["date"]   = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["year"]   = df["date"].dt.year.astype("Int64")
    df["month"]  = df["date"].dt.month.astype("Int64")
    df["season"] = df["month"].map(SEASON_MAP)
    df["primary_pollutant"] = (
        df["prominent_pollutants"].str.split(",").str[0].str.strip()
    )
    return df


# ── Shared helpers ────────────────────────────────────────────────────────────
def spines_off(ax):
    ax.spines[["top", "right"]].set_visible(False)


def kpi(col, value, label, sub=""):
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    """Custom styled metric card used on the Geographic page."""
    return f"""
    <div style='
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    '>
        <div style='font-size: 0.78rem; font-weight: 600; color: #64748b;
                    text-transform: uppercase; letter-spacing: 0.05em;
                    margin-bottom: 6px;'>{label}</div>
        <div style='font-size: 1.6rem; font-weight: 700; color: #0f172a;
                    line-height: 1.2;'>{value}</div>
    </div>
    """


def inject_css():
    """Call once at the top of every page file to apply shared styles."""
    st.markdown(CSS, unsafe_allow_html=True)