import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, inject_css, spines_off,
    STATUS_COLORS, STATUS_ORDER, SEASON_COLORS, MONTH_LABELS,
)

st.set_page_config(page_title="Trends | India AQI", page_icon="📈", layout="wide")
inject_css()

df = load_data()

st.markdown("""
<div class='page-hero'>
    <h1>📈 AQI Trends</h1>
    <p>How air quality has changed over time — monthly patterns, seasonal cycles,
    and year-over-year comparisons across India.</p>
</div>
""", unsafe_allow_html=True)

# ── Year-over-year comparison ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Year-over-Year Monthly AQI</div>",
            unsafe_allow_html=True)

yr_month = (
    df[df["year"].isin([2022, 2023, 2024])]
    .groupby(["year", "month"])["aqi_value"]
    .mean()
    .reset_index()
)
year_colors = {2022: "#f97316", 2023: "#6366f1", 2024: "#0ea5e9"}

fig, ax = plt.subplots(figsize=(12, 4))
for yr, grp in yr_month.groupby("year"):
    grp = grp.sort_values("month")
    ax.plot(grp["month"], grp["aqi_value"], marker="o", markersize=5,
            linewidth=2.2, label=str(int(yr)), color=year_colors.get(int(yr), "#888"))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=9)
ax.set_ylabel("Average AQI", fontsize=10)
ax.set_title("Monthly Average AQI by Year", fontweight="bold", fontsize=11)
ax.legend(title="Year", fontsize=9)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Seasonal analysis ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Seasonal AQI Distribution</div>",
            unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    season_aqi = df.groupby("season")["aqi_value"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(season_aqi.index, season_aqi.values,
                  color=[SEASON_COLORS[s] for s in season_aqi.index],
                  edgecolor="white", width=0.55)
    for bar, val in zip(bars, season_aqi.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean AQI", fontsize=10)
    ax.set_title("Average AQI by Season", fontweight="bold", fontsize=11)
    spines_off(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col_s2:
    season_status = (
        df.groupby(["season", "air_quality_status"])
        .size().reset_index(name="count")
    )
    season_total = df.groupby("season").size()
    season_status["pct"] = season_status.apply(
        lambda r: r["count"] / season_total[r["season"]] * 100, axis=1
    )
    pivot = season_status.pivot(
        index="season", columns="air_quality_status", values="pct"
    ).fillna(0)
    pivot = pivot.reindex(columns=[c for c in STATUS_ORDER if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(pivot))
    for status in pivot.columns:
        ax.bar(pivot.index, pivot[status], bottom=bottom,
               label=status, color=STATUS_COLORS[status], edgecolor="white", width=0.55)
        bottom += pivot[status].values
    ax.set_ylabel("Share (%)", fontsize=10)
    ax.set_title("Status Mix by Season", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    spines_off(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Monthly heatmap ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Month × Year AQI Heatmap</div>",
            unsafe_allow_html=True)

heat = (
    df[df["year"].isin([2022, 2023, 2024])]
    .groupby(["year", "month"])["aqi_value"]
    .mean()
    .unstack("month")
)
heat.columns = MONTH_LABELS
heat.index   = heat.index.astype(int)

fig, ax = plt.subplots(figsize=(12, 3))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, linecolor="#f8fafc", annot_kws={"size": 9})
ax.set_title("Mean Monthly AQI by Year", fontweight="bold", fontsize=11)
ax.set_xlabel("Month"); ax.set_ylabel("Year")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Status trend stacked area ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Status Share Over Time</div>",
            unsafe_allow_html=True)

period_status = (
    df.groupby([df["date"].dt.to_period("M"), "air_quality_status"])
    .size().reset_index(name="count")
)
period_status["date"] = period_status["date"].dt.to_timestamp()
period_total = period_status.groupby("date")["count"].sum()
period_status["pct"] = period_status.apply(
    lambda r: r["count"] / period_total[r["date"]] * 100, axis=1
)

fig, ax = plt.subplots(figsize=(12, 4))
pivot2 = period_status.pivot(
    index="date", columns="air_quality_status", values="pct"
).fillna(0)
pivot2 = pivot2.reindex(columns=[c for c in STATUS_ORDER if c in pivot2.columns])
pivot2.plot.area(ax=ax, color=[STATUS_COLORS[c] for c in pivot2.columns],
                 alpha=0.85, linewidth=0)
ax.set_ylabel("Share of Readings (%)", fontsize=10)
ax.set_xlabel("")
ax.set_title("Monthly Air Quality Status Share Over Time", fontweight="bold", fontsize=11)
ax.legend(loc="lower left", fontsize=8, ncol=6)
ax.set_ylim(0, 100)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Year-level summary table ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>Annual Summary</div>", unsafe_allow_html=True)

yr_summary = []
for yr in [2022, 2023, 2024]:
    sub = df[df["year"] == yr]
    yr_summary.append({
        "Year":              int(yr),
        "Readings":          f"{len(sub):,}",
        "Mean AQI":          f"{sub['aqi_value'].mean():.1f}",
        "Median AQI":        f"{sub['aqi_value'].median():.1f}",
        "Good / Satisf. %":  f"{(sub['air_quality_status'].isin(['Good','Satisfactory'])).mean()*100:.1f}%",
        "Poor / Worse %":    f"{(sub['air_quality_status'].isin(['Poor','Very Poor','Severe'])).mean()*100:.1f}%",
        "Worst Month (Avg)": MONTH_LABELS[sub.groupby('month')['aqi_value'].mean().idxmax() - 1],
    })
st.dataframe(pd.DataFrame(yr_summary), use_container_width=True, hide_index=True)

st.markdown("""
<div class='insight-box'>
📅 <strong>Seasonal pattern:</strong> Air quality in India deteriorates sharply in
<strong>October–January</strong> (post-monsoon + winter) — driven by crop stubble burning in Punjab/Haryana,
reduced wind speeds, and temperature inversions that trap pollutants close to the ground.
The Monsoon (June–September) consistently delivers the cleanest air nationally.
</div>
""", unsafe_allow_html=True)