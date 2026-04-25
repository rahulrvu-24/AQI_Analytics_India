import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_data, inject_css, spines_off, metric_card,
    STATUS_COLORS, STATUS_ORDER, MONTH_LABELS,
)

st.set_page_config(page_title="Geographic | India AQI", page_icon="🗺️", layout="wide")
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

ALL_STATES = sorted(df["state"].unique())

st.markdown("""
<div class='page-hero'>
    <h1>🗺️ Geographic Analysis</h1>
    <p>State-level and city-level rankings — which parts of India breathe the cleanest
    and most polluted air.</p>
</div>
""", unsafe_allow_html=True)

# ── State AQI rankings ────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>State Rankings by Average AQI</div>",
            unsafe_allow_html=True)

state_aqi = df.groupby("state")["aqi_value"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(state_aqi)))
bars = ax.barh(state_aqi.index[::-1], state_aqi.values[::-1],
               color=bar_colors[::-1], edgecolor="white", height=0.65)
for bar, val in zip(bars, state_aqi.values[::-1]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}", va="center", fontsize=8.5, color="#334155")
ax.set_xlabel("Mean AQI", fontsize=10)
ax.set_title("Average AQI by State (All Years)", fontweight="bold", fontsize=12)
ax.axvline(df["aqi_value"].mean(), color="#0ea5e9", linewidth=1.5,
           linestyle="--", label=f"National avg ({df['aqi_value'].mean():.0f})")
ax.legend(fontsize=9)
spines_off(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Top / Bottom 10 cities ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Most & Least Polluted Cities</div>",
            unsafe_allow_html=True)

city_aqi = df.groupby("area").agg(
    mean_aqi=("aqi_value", "mean"),
    readings=("aqi_value", "count"),
    state=("state", "first"),
).query("readings >= 50").sort_values("mean_aqi", ascending=False)

col_w, col_b = st.columns(2)

with col_w:
    st.markdown("**🔴 Top 10 Most Polluted**")
    top10 = city_aqi.head(10).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top10["area"][::-1], top10["mean_aqi"][::-1],
            color="#ef4444", edgecolor="white", height=0.6)
    for i, (_, row) in enumerate(top10[::-1].iterrows()):
        ax.text(row["mean_aqi"] + 1, i, f"{row['mean_aqi']:.0f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("Mean AQI"); spines_off(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col_b:
    st.markdown("**🟢 Top 10 Cleanest**")
    bot10 = city_aqi.tail(10).sort_values("mean_aqi").reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(bot10["area"], bot10["mean_aqi"],
            color="#22c55e", edgecolor="white", height=0.6)
    for i, (_, row) in enumerate(bot10.iterrows()):
        ax.text(row["mean_aqi"] + 0.5, i, f"{row['mean_aqi']:.0f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("Mean AQI"); spines_off(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── State deep-dive ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>State Deep-Dive</div>", unsafe_allow_html=True)

sel_state = st.selectbox(
    "Select a state", ALL_STATES,
    index=ALL_STATES.index("Delhi") if "Delhi" in ALL_STATES else 0,
)
state_df = df[df["state"] == sel_state]

# ── Custom metric cards ───────────────────────────────────────────────────────
mean_aqi    = f"{state_df['aqi_value'].mean():.1f}"
readings    = f"{len(state_df):,}"
cities      = str(state_df["area"].nunique())
worst_month = MONTH_LABELS[state_df.groupby("month")["aqi_value"].mean().idxmax() - 1]

s1, s2, s3, s4 = st.columns(4)
s1.markdown(metric_card("Mean AQI",    mean_aqi),    unsafe_allow_html=True)
s2.markdown(metric_card("Readings",    readings),    unsafe_allow_html=True)
s3.markdown(metric_card("Cities",      cities),      unsafe_allow_html=True)
s4.markdown(metric_card("Worst Month", worst_month), unsafe_allow_html=True)

st.write("")
st.write("")

# ── Monthly AQI + City Rankings ───────────────────────────────────────────────
col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    monthly_s = state_df.groupby("month")["aqi_value"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.fill_between(range(1, 13), [monthly_s.get(m, np.nan) for m in range(1, 13)],
                    alpha=0.15, color="#6366f1")
    ax.plot(range(1, 13), [monthly_s.get(m, np.nan) for m in range(1, 13)],
            color="#6366f1", linewidth=2, marker="o", markersize=5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_ylabel("Mean AQI")
    ax.set_title(f"{sel_state} — Monthly AQI", fontweight="bold", fontsize=10)
    spines_off(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col_sc2:
    city_rank = (
        state_df.groupby("area")["aqi_value"]
        .mean().sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(city_rank.index[::-1], city_rank.values[::-1],
            color="#6366f1", edgecolor="white", height=0.6)
    ax.set_xlabel("Mean AQI")
    ax.set_title(f"{sel_state} — City Rankings", fontweight="bold", fontsize=10)
    spines_off(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Status breakdown ──────────────────────────────────────────────────────────
st_status = (
    state_df["air_quality_status"]
    .value_counts()
    .reindex(STATUS_ORDER)
    .dropna()
)

st.markdown(f"**Status breakdown for {sel_state}:**")
cols_st = st.columns(len(st_status))
for col_i, (status, count) in zip(cols_st, st_status.items()):
    col_i.markdown(
        f"<div style='text-align:center; padding:10px; border-radius:8px; "
        f"background:{STATUS_COLORS[status]}22; border:1px solid {STATUS_COLORS[status]}55;'>"
        f"<div style='font-weight:700; color:{STATUS_COLORS[status]};font-size:0.9rem;'>{status}</div>"
        f"<div style='font-size:1.2rem;font-weight:700; color:#ffffff;'>{count:,}</div>"
        f"<div style='font-size:0.75rem;color:#94a3b8;'>{count/len(state_df)*100:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )