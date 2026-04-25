import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_data, inject_css, kpi, spines_off,
    STATUS_COLORS, STATUS_ORDER,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar info ──────────────────────────────────────────────────────────────
df = load_data()
ALL_STATES = sorted(df["state"].unique())
ALL_YEARS = sorted(df["year"].dropna().unique().astype(int).tolist())
ALL_POLLUTANTS = sorted(df["primary_pollutant"].unique())

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

# ── Page content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-hero'>
    <h1>🌿 India Air Quality Analytics</h1>
    <p>Three years of CPCB monitoring data — 235K daily readings across 32 states and 291 cities,
    from April 2022 to April 2025.</p>
</div>
""", unsafe_allow_html=True)

# ── Top KPIs ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, f"{len(df):,}",                       "Total Readings",    "Apr 2022 – Apr 2025")
kpi(c2, f"{df['state'].nunique()}",            "States",            "All of India")
kpi(c3, f"{df['area'].nunique()}",             "Cities",            "291 monitoring sites")
kpi(c4, f"{df['aqi_value'].mean():.0f}",       "Avg AQI",           "National mean")
good_pct = (df["air_quality_status"].isin(["Good", "Satisfactory"])).mean() * 100
kpi(c5, f"{good_pct:.1f}%",                   "Good/Satisfactory", "of all readings")

# ── Status breakdown donut ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Air Quality Status Distribution</div>",
            unsafe_allow_html=True)

col_d, col_t = st.columns([1, 1])

with col_d:
    counts = df["air_quality_status"].value_counts().reindex(STATUS_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    wedge_colors = [STATUS_COLORS[s] for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")
    ax.set_title("All Readings — Status Share", fontweight="bold", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col_t:
    st.markdown("<br>", unsafe_allow_html=True)
    total = len(df)
    for status in STATUS_ORDER:
        n   = counts.get(status, 0)
        pct = n / total * 100
        col = STATUS_COLORS[status]
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; "
            f"padding:6px 0; border-bottom:1px solid #f1f5f9;'>"
            f"<div style='background:{col}; width:12px; height:12px; border-radius:3px; flex-shrink:0;'></div>"
            f"<div style='flex:1; font-weight:500; color:#1e293b; font-size:0.9rem;'>{status}</div>"
            f"<div style='color:#64748b; font-size:0.85rem;'>{n:,}</div>"
            f"<div style='width:80px; text-align:right; font-weight:600; font-size:0.9rem; color:{col};'>{pct:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── National monthly AQI trend ────────────────────────────────────────────────
st.markdown("<div class='section-title'>National AQI Trend — Monthly Average</div>",
            unsafe_allow_html=True)

monthly = (
    df.groupby(df["date"].dt.to_period("M"))["aqi_value"]
    .mean().reset_index()
)
monthly["date"] = monthly["date"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.fill_between(monthly["date"], monthly["aqi_value"], alpha=0.12, color="#0ea5e9")
ax.plot(monthly["date"], monthly["aqi_value"], color="#0ea5e9", linewidth=2)
for (lo, hi, col, lbl) in [
    (0,   50,  "#00C853", "Good"),
    (51,  100, "#8BC34A", "Satisfactory"),
    (101, 200, "#FFD600", "Moderate"),
]:
    ax.axhspan(lo, hi, alpha=0.06, color=col)
ax.set_ylabel("Average AQI", fontsize=10)
ax.set_xlabel("")
ax.set_title("India Monthly Average AQI (Apr 2022 – Apr 2025)", fontweight="bold", fontsize=11)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Key insights ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Key Findings</div>", unsafe_allow_html=True)

worst_state = df.groupby("state")["aqi_value"].mean().idxmax()
best_state  = df.groupby("state")["aqi_value"].mean().idxmin()
worst_city  = df.groupby("area")["aqi_value"].mean().idxmax()
severe_pct  = (df["air_quality_status"] == "Severe").mean() * 100
pm10_pct    = (df["primary_pollutant"] == "PM10").mean() * 100

st.markdown(f"""
<div class='warn-box'>
🔴 <strong>{worst_state}</strong> is India's most polluted state by average AQI.
<strong>{worst_city}</strong> is the most polluted city in the dataset.
Severe air quality accounts for <strong>{severe_pct:.2f}%</strong> of all readings.
</div>
<div class='good-box'>
🟢 <strong>{best_state}</strong> has the lowest average AQI — cleanest air among all monitored states.
<strong>{good_pct:.1f}%</strong> of all readings fall in the Good or Satisfactory range.
</div>
<div class='insight-box'>
🌬️ <strong>PM10</strong> is the dominant pollutant, appearing in <strong>{pm10_pct:.1f}%</strong> of primary
pollutant readings — driven by road dust, construction, and desert dust in northern and western India.
Winter months (Oct–Jan) consistently record the worst air quality nationwide.
</div>
""", unsafe_allow_html=True)