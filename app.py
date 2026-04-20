# TO RUN: python -m streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

SEASON_MAP   = {1:"Winter",2:"Winter",3:"Pre-monsoon",4:"Pre-monsoon",5:"Pre-monsoon",
                6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Post-monsoon",
                10:"Post-monsoon",11:"Post-monsoon",12:"Winter"}
SEASON_COLORS = {"Winter":"#90CAF9","Pre-monsoon":"#A5D6A7",
                 "Monsoon":"#80DEEA","Post-monsoon":"#FFCC80"}
MONTH_LABELS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


# ── Data loading & caching ────────────────────────────────────────────────────
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
    df["primary_pollutant"] = df["prominent_pollutants"].str.split(",").str[0].str.strip()
    return df


def spines_off(ax):
    ax.spines[["top", "right"]].set_visible(False)


def kpi(col, value, label, sub=""):
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


df = load_data()
ALL_STATES    = sorted(df["state"].unique())
ALL_YEARS     = sorted(df["year"].dropna().unique().astype(int).tolist())
ALL_POLLUTANTS = sorted(df["primary_pollutant"].unique())


# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    st.markdown("<div class='nav-label'>Pages</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📈  Trends",
        "🗺️  Geographic",
        "🌬️  Pollutants",
        "🔍  Explorer",
    ], label_visibility="collapsed")

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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class='page-hero'>
        <h1>🌿 India Air Quality Analytics</h1>
        <p>Three years of CPCB monitoring data — 235K daily readings across 32 states and 291 cities,
        from April 2022 to April 2025.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{len(df):,}",       "Total Readings",      "Apr 2022 – Apr 2025")
    kpi(c2, f"{df['state'].nunique()}",  "States",       "All of India")
    kpi(c3, f"{df['area'].nunique()}",   "Cities",       "291 monitoring sites")
    kpi(c4, f"{df['aqi_value'].mean():.0f}", "Avg AQI",  "National mean")
    good_pct = (df["air_quality_status"].isin(["Good","Satisfactory"])).mean() * 100
    kpi(c5, f"{good_pct:.1f}%",   "Good/Satisfactory",   "of all readings")

    # ── Status breakdown donut ─────────────────────────────────────────────────
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

    # ── National monthly AQI trend ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>National AQI Trend — Monthly Average</div>",
                unsafe_allow_html=True)

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["aqi_value"]
        .mean()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.fill_between(monthly["date"], monthly["aqi_value"], alpha=0.12, color="#0ea5e9")
    ax.plot(monthly["date"], monthly["aqi_value"], color="#0ea5e9", linewidth=2)

    # Shade AQI bands
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

    # ── Key insights ──────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Trends":
    st.markdown("""
    <div class='page-hero'>
        <h1>📈 AQI Trends</h1>
        <p>How air quality has changed over time — monthly patterns, seasonal cycles,
        and year-over-year comparisons across India.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Year-over-year comparison ─────────────────────────────────────────────
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

    # ── Seasonal analysis ─────────────────────────────────────────────────────
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
        pivot = season_status.pivot(index="season", columns="air_quality_status", values="pct").fillna(0)
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

    # ── Monthly heatmap ───────────────────────────────────────────────────────
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

    # ── Status trend stacked area ─────────────────────────────────────────────
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
    pivot2 = period_status.pivot(index="date", columns="air_quality_status", values="pct").fillna(0)
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

    # ── Year-level summary table ───────────────────────────────────────────────
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

    st.markdown(f"""
    <div class='insight-box'>
    📅 <strong>Seasonal pattern:</strong> Air quality in India deteriorates sharply in
    <strong>October–January</strong> (post-monsoon + winter) — driven by crop stubble burning in Punjab/Haryana,
    reduced wind speeds, and temperature inversions that trap pollutants close to the ground.
    The Monsoon (June–September) consistently delivers the cleanest air nationally.
    </div>
    """, unsafe_allow_html=True)


# PAGE 3 — GEOGRAPHIC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  Geographic":
    st.markdown("""
    <div class='page-hero'>
        <h1>🗺️ Geographic Analysis</h1>
        <p>State-level and city-level rankings — which parts of India breathe the cleanest
        and most polluted air.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── State AQI rankings ────────────────────────────────────────────────────
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

    # ── Top / Bottom 10 cities ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Most & Least Polluted Cities</div>",
                unsafe_allow_html=True)

    city_aqi = df.groupby("area").agg(
        mean_aqi=("aqi_value","mean"),
        readings=("aqi_value","count"),
        state=("state","first"),
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

    # ── State deep-dive ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>State Deep-Dive</div>", unsafe_allow_html=True)

    sel_state = st.selectbox("Select a state", ALL_STATES,
                             index=ALL_STATES.index("Delhi") if "Delhi" in ALL_STATES else 0)
    state_df  = df[df["state"] == sel_state]

    # ── Custom metric cards ────────────────────────────────────────────────
    mean_aqi    = f"{state_df['aqi_value'].mean():.1f}"
    readings    = f"{len(state_df):,}"
    cities      = str(state_df["area"].nunique())
    worst_month = MONTH_LABELS[state_df.groupby("month")["aqi_value"].mean().idxmax() - 1]

    def metric_card(label, value):
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

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(metric_card("Mean AQI",    mean_aqi),    unsafe_allow_html=True)
    s2.markdown(metric_card("Readings",    readings),    unsafe_allow_html=True)
    s3.markdown(metric_card("Cities",      cities),      unsafe_allow_html=True)
    s4.markdown(metric_card("Worst Month", worst_month), unsafe_allow_html=True)

    st.write("")   # ← add here
    st.write("")   # ← add here

    # ── Monthly AQI + City Rankings ───────────────────────────────────────
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
        st.pyplot(fig)
        plt.close()
    with col_sc2:
        city_rank = (state_df.groupby("area")["aqi_value"]
                     .mean()
                     .sort_values(ascending=False)
                     .head(10))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(city_rank.index[::-1], city_rank.values[::-1],
                color="#6366f1", edgecolor="white", height=0.6)
        ax.set_xlabel("Mean AQI")
        ax.set_title(f"{sel_state} — City Rankings", fontweight="bold", fontsize=10)
        spines_off(ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
   # ── Status breakdown ──────────────────────────────────────────────────
    st_status = (state_df["air_quality_status"]
                 .value_counts()
                 .reindex(STATUS_ORDER)
                 .dropna())

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
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POLLUTANTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌬️  Pollutants":
    st.markdown("""
    <div class='page-hero'>
        <h1>🌬️ Pollutant Analysis</h1>
        <p>Which pollutants dominate India's air, how they correlate with air quality severity,
        and how their prevalence varies by state and season.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overall pollutant share ───────────────────────────────────────────────
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

    # ── Pollutant × mean AQI ─────────────────────────────────────────────────
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

    # ── Pollutant × Status heatmap ────────────────────────────────────────────
    st.markdown("<div class='section-title'>Pollutant × Air Quality Status Heatmap</div>",
                unsafe_allow_html=True)

    heat_df = df[df["primary_pollutant"].isin(top_poll.head(8).index)]
    pivot = (
        heat_df.groupby(["primary_pollutant","air_quality_status"])
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

    # ── Pollutant by season ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Pollutant Prevalence by Season</div>",
                unsafe_allow_html=True)

    season_poll = (
        df[df["primary_pollutant"].isin(top_poll.head(6).index)]
        .groupby(["season","primary_pollutant"])
        .size().reset_index(name="count")
    )
    season_total_poll = df[df["primary_pollutant"].isin(top_poll.head(6).index)].groupby("season").size()
    season_poll["pct"] = season_poll.apply(
        lambda r: r["count"] / season_total_poll[r["season"]] * 100, axis=1
    )

    pivot_sp = season_poll.pivot(index="season", columns="primary_pollutant", values="pct").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot_sp.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
    ax.set_xlabel("Season"); ax.set_ylabel("Share of Readings (%)")
    ax.set_title("Pollutant Share by Season", fontweight="bold", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Pollutant", fontsize=8, loc="upper right")
    spines_off(ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── State × Pollutant heatmap ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Dominant Pollutant by State</div>",
                unsafe_allow_html=True)

    state_poll = (
        df[df["primary_pollutant"].isin(top_poll.head(6).index)]
        .groupby(["state","primary_pollutant"])
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

    # ── Insights ─────────────────────────────────────────────────────────────
    pm10_aqi = df[df["primary_pollutant"]=="PM10"]["aqi_value"].mean()
    o3_aqi   = df[df["primary_pollutant"]=="O3"]["aqi_value"].mean()
    co_aqi   = df[df["primary_pollutant"]=="CO"]["aqi_value"].mean()

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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Explorer":
    st.markdown("""
    <div class='page-hero'>
        <h1>🔍 Data Explorer</h1>
        <p>Filter by state, year, and pollutant to drill into any slice of the dataset
        and compare custom subsets side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
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

    # ── Filtered KPIs ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Filtered Summary</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, f"{len(fdf):,}",              "Readings")
    kpi(k2, f"{fdf['aqi_value'].mean():.1f}", "Mean AQI")
    kpi(k3, f"{fdf['aqi_value'].median():.1f}","Median AQI")
    kpi(k4, f"{fdf['aqi_value'].max():.0f}",  "Peak AQI")
    good_f = fdf["air_quality_status"].isin(["Good","Satisfactory"]).mean()*100
    kpi(k5, f"{good_f:.1f}%",             "Good/Satisf.")

    # ── AQI distribution ──────────────────────────────────────────────────────
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

    # ── State comparison (if multiple selected) ───────────────────────────────
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

    # ── Monthly trend for selection ───────────────────────────────────────────
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

    # ── Raw data preview ──────────────────────────────────────────────────────
    with st.expander(f"📋 Preview filtered data ({len(fdf):,} rows)"):
        st.dataframe(
            fdf[["date","state","area","primary_pollutant","aqi_value",
                 "air_quality_status","number_of_monitoring_stations"]]
            .sort_values("date", ascending=False)
            .head(500)
            .reset_index(drop=True),
            use_container_width=True,
        )
        st.caption("Showing up to 500 rows. Use filters above to narrow down.")
