import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, inject_css, spines_off,
    STATUS_COLORS, STATUS_ORDER,
)

st.set_page_config(page_title="Pollutants | India AQI", page_icon="🌬️", layout="wide")
inject_css()

df = load_data()

st.markdown("""
<div class='page-hero'>
    <h1>🌬️ Pollutant Analysis</h1>
    <p>Which pollutants dominate India's air, how they correlate with air quality severity,
    and how their prevalence varies by state and season.</p>
</div>
""", unsafe_allow_html=True)

# ── Overall pollutant share ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>Primary Pollutant Frequency</div>",
            unsafe_allow_html=True)

top_poll = df["primary_pollutant"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 4))
colors_p = ["#6366f1" if i == 0 else "#a5b4fc" for i in range(len(top_poll))]
bars = ax.bar(top_poll.index, top_poll.values, color=colors_p, edgecolor="white", width=0.6)
for bar, val in zip(bars, top_poll.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
            f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Readings")
ax.set_title("Top 10 Primary Pollutants by Frequency", fontweight="bold", fontsize=12)
ax.tick_params(axis="x", rotation=15)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Pollutant × mean AQI ─────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Average AQI per Pollutant</div>",
            unsafe_allow_html=True)

poll_aqi = (
    df[df["primary_pollutant"].isin(top_poll.index)]
    .groupby("primary_pollutant")["aqi_value"]
    .mean().sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(poll_aqi.index, poll_aqi.values,
              color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(poll_aqi))),
              edgecolor="white", width=0.6)
for bar, val in zip(bars, poll_aqi.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Mean AQI")
ax.set_title("Mean AQI by Primary Pollutant", fontweight="bold", fontsize=12)
ax.tick_params(axis="x", rotation=15)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Pollutant × Status heatmap ────────────────────────────────────────────────
st.markdown("<div class='section-title'>Pollutant × Air Quality Status Heatmap</div>",
            unsafe_allow_html=True)

heat_df = df[df["primary_pollutant"].isin(top_poll.head(8).index)]
pivot = (
    heat_df.groupby(["primary_pollutant", "air_quality_status"])
    .size().unstack(fill_value=0)
)
pivot = pivot.reindex(columns=[c for c in STATUS_ORDER if c in pivot.columns])

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
            linewidths=0.4, linecolor="#f8fafc", annot_kws={"size": 9})
ax.set_title("Pollutant × Status Count", fontweight="bold", fontsize=12)
ax.set_xlabel("Air Quality Status"); ax.set_ylabel("Primary Pollutant")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Pollutant by season ───────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Pollutant Prevalence by Season</div>",
            unsafe_allow_html=True)

season_poll = (
    df[df["primary_pollutant"].isin(top_poll.head(6).index)]
    .groupby(["season", "primary_pollutant"])
    .size().reset_index(name="count")
)
season_total_poll = (
    df[df["primary_pollutant"].isin(top_poll.head(6).index)]
    .groupby("season").size()
)
season_poll["pct"] = season_poll.apply(
    lambda r: r["count"] / season_total_poll[r["season"]] * 100, axis=1
)

pivot_sp = season_poll.pivot(
    index="season", columns="primary_pollutant", values="pct"
).fillna(0)
fig, ax = plt.subplots(figsize=(10, 4.5))
pivot_sp.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
ax.set_xlabel("Season"); ax.set_ylabel("Share of Readings (%)")
ax.set_title("Pollutant Share by Season", fontweight="bold", fontsize=12)
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Pollutant", fontsize=8, loc="upper right")
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── State × Pollutant heatmap ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Dominant Pollutant by State</div>",
            unsafe_allow_html=True)

state_poll = (
    df[df["primary_pollutant"].isin(top_poll.head(6).index)]
    .groupby(["state", "primary_pollutant"])
    .size().unstack(fill_value=0)
)
state_poll_pct = state_poll.div(state_poll.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(state_poll_pct, annot=True, fmt=".0f", cmap="Blues", ax=ax,
            linewidths=0.3, linecolor="#f8fafc", annot_kws={"size": 8})
ax.set_title("% of Readings per Pollutant by State", fontweight="bold", fontsize=12)
ax.set_xlabel("Primary Pollutant"); ax.set_ylabel("State")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Insights ──────────────────────────────────────────────────────────────────
pm10_aqi = df[df["primary_pollutant"] == "PM10"]["aqi_value"].mean()
o3_aqi   = df[df["primary_pollutant"] == "O3"]["aqi_value"].mean()
co_aqi   = df[df["primary_pollutant"] == "CO"]["aqi_value"].mean()

st.markdown(f"""
<div class='insight-box'>
🔬 <strong>PM10</strong> dominates India's air quality problem ({top_poll['PM10']:,} readings,
mean AQI {pm10_aqi:.0f}) — driven by road dust, construction activity, and arid/semi-arid geography
in Rajasthan, UP, and Haryana.<br><br>
<strong>O3 (Ozone)</strong> is a photochemical pollutant — most prevalent in Pre-monsoon/Summer
when UV radiation is highest (mean AQI {o3_aqi:.0f}). <strong>CO</strong> peaks in Winter
due to vehicular exhaust and biomass burning (mean AQI {co_aqi:.0f}).
</div>
""", unsafe_allow_html=True)