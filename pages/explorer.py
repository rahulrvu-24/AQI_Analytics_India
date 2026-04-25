import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_data, inject_css, kpi, spines_off,
    STATUS_COLORS, STATUS_ORDER,
)

st.set_page_config(page_title="Explorer | India AQI", page_icon="🔍", layout="wide")
inject_css()

df = load_data()

with st.sidebar:
    st.markdown("""
    <div style='padding:18px 4px 8px;'>
        <div style='font-size:1.6rem;'>🌿</div>
        <div style='font-size:1.1rem; font-weight:700; color:#f1f5f9 !important; margin-top:4px;'>
            India AQI Analytics
        </div>
        <div style='font-size:0.76rem; color:#94a3b8 !important; margin-top:2px;'>
            CPCB · Apr 2022 – Apr 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<hr style='border-color:rgba(255,255,255,0.07); margin:16px 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-label'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:0.85rem;'>"
        f"📋 {len(df):,} records<br>"
        f"🗺️ {df['state'].nunique()} states<br>"
        f"🏙️ {df['area'].nunique()} cities<br>"
        f"📅 {df['year'].min()}–{df['year'].max()}"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top:24px; font-size:0.7rem; color:#475569 !important;'>"
        "Source: CPCB India · Built with Streamlit</div>",
        unsafe_allow_html=True,
    )
    
ALL_STATES     = sorted(df["state"].unique())
ALL_YEARS      = sorted(df["year"].dropna().unique().astype(int).tolist())
ALL_POLLUTANTS = sorted(df["primary_pollutant"].unique())

st.markdown("""
<div class='page-hero'>
    <h1>🔍 Data Explorer</h1>
    <p>Filter by state, year, and pollutant to drill into any slice of the dataset
    and compare custom subsets side by side.</p>
</div>
""", unsafe_allow_html=True)

# ── Filters ───────────────────────────────────────────────────────────────────
with st.container():
    fc1, fc2, fc3 = st.columns(3)
    sel_states = fc1.multiselect(
        "States", ALL_STATES,
        default=["Delhi", "Maharashtra"] if "Delhi" in ALL_STATES else ALL_STATES[:2],
    )
    sel_years = fc2.multiselect(
        "Years", ALL_YEARS, default=ALL_YEARS,
    )
    sel_polls = fc3.multiselect(
        "Primary Pollutants", ALL_POLLUTANTS,
        default=["PM10", "PM2.5"] if "PM10" in ALL_POLLUTANTS else ALL_POLLUTANTS[:2],
    )

if not sel_states or not sel_years or not sel_polls:
    st.warning("Please select at least one state, year, and pollutant.")
    st.stop()

fdf = df[
    df["state"].isin(sel_states) &
    df["year"].isin(sel_years) &
    df["primary_pollutant"].isin(sel_polls)
]

if fdf.empty:
    st.error("No data matches this filter combination.")
    st.stop()

# ── Filtered KPIs ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Filtered Summary</div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, f"{len(fdf):,}",                          "Readings")
kpi(k2, f"{fdf['aqi_value'].mean():.1f}",          "Mean AQI")
kpi(k3, f"{fdf['aqi_value'].median():.1f}",        "Median AQI")
kpi(k4, f"{fdf['aqi_value'].max():.0f}",           "Peak AQI")
good_f = fdf["air_quality_status"].isin(["Good", "Satisfactory"]).mean() * 100
kpi(k5, f"{good_f:.1f}%",                          "Good/Satisf.")

# ── AQI distribution ──────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>AQI Distribution</div>", unsafe_allow_html=True)

col_h, col_b = st.columns(2)

with col_h:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fdf["aqi_value"], bins=40, color="#6366f1", edgecolor="white", alpha=0.85)
    ax.axvline(fdf["aqi_value"].mean(), color="#ef4444", linewidth=1.5,
               linestyle="--", label=f"Mean {fdf['aqi_value'].mean():.0f}")
    ax.axvline(fdf["aqi_value"].median(), color="#f97316", linewidth=1.5,
               linestyle=":", label=f"Median {fdf['aqi_value'].median():.0f}")
    ax.set_xlabel("AQI Value"); ax.set_ylabel("Count")
    ax.set_title("AQI Value Distribution", fontweight="bold")
    ax.legend(fontsize=8)
    spines_off(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

with col_b:
    st_counts = fdf["air_quality_status"].value_counts().reindex(STATUS_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(st_counts.index, st_counts.values,
           color=[STATUS_COLORS[s] for s in st_counts.index],
           edgecolor="white", width=0.6)
    ax.set_ylabel("Readings"); ax.set_title("Status Counts", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    spines_off(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

# ── State comparison (if multiple selected) ───────────────────────────────────
if len(sel_states) > 1:
    st.markdown("<div class='section-title'>State Comparison</div>",
                unsafe_allow_html=True)

    state_comp = fdf.groupby("state")["aqi_value"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, len(sel_states) * 0.7)))
    palette = plt.cm.tab10(np.linspace(0, 1, len(state_comp)))
    ax.barh(state_comp.index[::-1], state_comp.values[::-1],
            color=palette[::-1], edgecolor="white", height=0.6)
    for i, (s, v) in enumerate(zip(state_comp.index[::-1], state_comp.values[::-1])):
        ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Mean AQI")
    ax.set_title("Mean AQI — Selected States", fontweight="bold")
    spines_off(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Monthly trend for selection ───────────────────────────────────────────────
st.markdown("<div class='section-title'>Monthly AQI Trend (Filtered)</div>",
            unsafe_allow_html=True)

trend = (
    fdf.groupby(fdf["date"].dt.to_period("M"))["aqi_value"]
    .mean().reset_index()
)
trend["date"] = trend["date"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.fill_between(trend["date"], trend["aqi_value"], alpha=0.12, color="#6366f1")
ax.plot(trend["date"], trend["aqi_value"], color="#6366f1", linewidth=2)
ax.set_ylabel("Mean AQI"); ax.set_xlabel("")
ax.set_title("Monthly Average AQI — Filtered Selection", fontweight="bold")
spines_off(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Raw data preview ──────────────────────────────────────────────────────────
with st.expander(f"📋 Preview filtered data ({len(fdf):,} rows)"):
    st.dataframe(
        fdf[["date", "state", "area", "primary_pollutant", "aqi_value",
             "air_quality_status", "number_of_monitoring_stations"]]
        .sort_values("date", ascending=False)
        .head(500)
        .reset_index(drop=True),
        use_container_width=True,
    )
    st.caption("Showing up to 500 rows. Use filters above to narrow down.")