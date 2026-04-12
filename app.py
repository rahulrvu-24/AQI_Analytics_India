# TO RUN: python -m streamlit run app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import model as M

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Air Quality Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1b2a 0%, #1b2d42 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem;
    padding: 6px 0;
}

/* ── Main area ── */
.main .block-container { padding-top: 2rem; max-width: 1100px; }

/* ── Page title ── */
.page-hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    color: white;
}
.page-hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px; color: white; }
.page-hero p  { font-size: 1rem; margin: 0; opacity: 0.88; color: white; }

/* ── Section header ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    letter-spacing: -0.02em;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Stat card ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0f172a; }

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 999px;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    letter-spacing: 0.01em;
    margin: 8px 0 4px;
}

/* ── Info / warning boxes ── */
.info-card {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.92rem;
    margin: 10px 0;
    color: #0c4a6e;
}
.warn-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.92rem;
    margin: 10px 0;
    color: #78350f;
}
.insight-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}

/* ── AQI pill inline ── */
.aqi-pill {
    display:inline-block; padding:3px 12px; border-radius:999px;
    font-size:0.82rem; font-weight:600; color:white; margin-left:6px;
}

/* ── Predict result panel ── */
.result-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 12px;
}

/* ── Pipeline step ── */
.step-row {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 10px 0; border-bottom: 1px solid #f1f5f9;
}
.step-num {
    background: #6366f1; color: white;
    border-radius: 8px; width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.step-text b { font-weight: 600; color: #1e293b; }
.step-text span { color: #64748b; font-size: 0.9rem; }

/* ── Code blocks ── */
code { font-family: 'DM Mono', monospace; }

/* ── Data table ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── Sidebar nav label ── */
.nav-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569 !important; margin: 14px 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached data + model ───────────────────────────────────────────────────────
@st.cache_data
def get_raw_data():
    return M.load_data(M.DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(M.MODEL_FILE) and os.path.exists(M.ENCODER_FILE):
        artifacts = joblib.load(M.MODEL_FILE)
        encoders  = joblib.load(M.ENCODER_FILE)
        return artifacts["model"], artifacts["le_target"], encoders, None

    df = M.load_data()
    df = M.preprocess(df)
    df = M.engineer_features(df)
    df = M.encode_features(df, fit=True)
    model, X_test, y_test, le_target = M.train(df)
    results  = M.evaluate(model, X_test, y_test, le_target)
    encoders = joblib.load(M.ENCODER_FILE)
    return model, le_target, encoders, results


def get_eval_results(model, le_target):
    df = M.load_data()
    df = M.preprocess(df)
    df = M.engineer_features(df)
    df = M.encode_features(df, fit=False)
    y  = le_target.transform(df[M.TARGET_COL])
    X  = df[M.FEATURE_COLS]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return M.evaluate(model, X_test, y_test, le_target)


# ── Sidebar ───────────────────────────────────────────────────────────────────
model_ready = os.path.exists(M.MODEL_FILE) and os.path.exists(M.ENCODER_FILE)

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 4px 8px;'>
        <div style='font-size:1.7rem;'>🌿</div>
        <div style='font-size:1.15rem; font-weight:700; color:#f1f5f9 !important; margin-top:4px;'>
            AQI Classifier
        </div>
        <div style='font-size:0.78rem; color:#94a3b8 !important; margin-top:2px;'>
            India Air Quality · SVM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔮  Predict", "📊  EDA", "🤖  Model Metrics"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>Dataset</div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.88rem;'>India CPCB · 235,785 records · 32 states</span>", unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>Model</div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.88rem;'>SVC · RBF kernel · C=10 · balanced weights</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:16px 0;'>", unsafe_allow_html=True)

    if model_ready:
        st.markdown(
            "<div style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); "
            "border-radius:8px; padding:8px 12px; font-size:0.82rem; color:#6ee7b7 !important;'>"
            "✅ Model loaded from disk</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.3); "
            "border-radius:8px; padding:8px 12px; font-size:0.82rem; color:#fde68a !important;'>"
            "⚙️ Model trains on first run (2–5 min)</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='margin-top:24px; font-size:0.72rem; color:#475569 !important;'>"
        "Built with Streamlit · scikit-learn</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='page-hero'>
        <h1>🌿 AQI Air Quality Classifier</h1>
        <p>Predicting India's air quality status using Support Vector Machine (SVM) —
        trained on real CPCB monitoring data across 32 states.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        st.markdown("""
        <div class='warn-card'>
        ⚙️ <strong>First launch detected.</strong> The SVM will be trained automatically when you
        visit <strong>Predict</strong> or <strong>Model Metrics</strong>. Training takes 2–5 minutes,
        after which the model is saved to <code>svm_model.pkl</code> and loads instantly every time.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-card'>
        ✅ <strong>Model ready.</strong> <code>svm_model.pkl</code> found on disk — predictions are instant.
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📋 Dataset at a Glance</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  "2,35,785")
    c2.metric("States Covered", "32")
    c3.metric("Cities / Areas", "500+")
    c4.metric("Target Classes", "6")

    # ── AQI categories ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎨 AQI Status Categories</div>", unsafe_allow_html=True)
    for status, (lo, hi, desc) in M.AQI_CATEGORY_INFO.items():
        color = M.STATUS_COLORS[status]
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:14px; padding:9px 0;
                    border-bottom:1px solid #f1f5f9;'>
            <div style='background:{color}; min-width:120px; text-align:center;
                        padding:5px 14px; border-radius:999px; color:white;
                        font-weight:700; font-size:0.82rem;'>{status}</div>
            <div style='color:#475569; font-size:0.9rem;'>
                <strong style='color:#1e293b;'>AQI {lo}–{hi}</strong> — {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ML Pipeline ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🔄 ML Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Load Data",            "Read 235K rows from aqi.csv"),
        ("2", "Preprocess",           "Drop nulls, parse dates, strip whitespace"),
        ("3", "Feature Engineering",  "Extract primary pollutant, multi-pollutant flag"),
        ("4", "Encode",               "Label-encode state, area, pollutants"),
        ("5", "Scale",                "StandardScaler — zero-mean, unit-variance (required by SVM)"),
        ("6", "Train SVM",            "SVC · RBF kernel · C=10 · class_weight='balanced'"),
        ("7", "Save",                 "Persist svm_model.pkl + encoders.pkl to disk"),
        ("8", "Evaluate",             "80/20 stratified split · accuracy, F1, confusion matrix"),
        ("9", "Predict",              "Inference via app — model loaded from disk automatically"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class='step-row'>
            <div class='step-num'>{num}</div>
            <div class='step-text'><b>{title}</b><br><span>{desc}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Why SVM ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🤖 Why SVM?</div>", unsafe_allow_html=True)
    st.markdown("""
    | Property | Detail |
    |---|---|
    | **Kernel** | RBF — captures non-linear boundaries between AQI classes |
    | **Scaling** | StandardScaler applied inside a Pipeline |
    | **Class imbalance** | `class_weight='balanced'` handles rare Severe class (555 rows) |
    | **Confidence** | `probability=True` enables soft probability scores |
    | **Feature leakage fix** | `aqi_value` & `aqi_bucket` excluded — they directly encode the target |
    """)

    # ── Project structure ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🗂️ Project Structure</div>", unsafe_allow_html=True)
    st.code("""
AQI_Project/
├── aqi.csv          ← Raw dataset (India CPCB, 235K rows)
├── model.py         ← Full ML pipeline (preprocess → SVM train → predict)
├── app.py           ← This Streamlit app (auto-trains on first run)
└── svm_model.pkl    ← Saved model (auto-generated)
    """, language="")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict":
    st.markdown("""
    <div class='page-hero'>
        <h1>🔮 Predict Air Quality Status</h1>
        <p>The SVM predicts air quality from location, pollutant type, and time —
        without relying on the AQI value (which would leak the answer).</p>
    </div>
    """, unsafe_allow_html=True)

    df_raw     = get_raw_data()
    states     = sorted(df_raw["state"].unique())
    pollutants = sorted(df_raw["prominent_pollutants"].unique())

    if not model_ready:
        st.markdown("""
        <div class='warn-card'>
        ⚙️ <strong>Training SVM for the first time…</strong>
        This takes 2–5 minutes. The model is saved automatically — future launches are instant.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Loading model (training on first launch — please wait)..."):
        model, le_target, encoders, _ = get_model()

    st.markdown(
        "<div class='info-card' style='margin-bottom:20px;'>✅ Model ready for prediction.</div>",
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.container():
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='section-title'>📍 Location</div>", unsafe_allow_html=True)
            state = st.selectbox(
                "State",
                states,
                index=states.index("Delhi") if "Delhi" in states else 0,
            )
            areas = sorted(df_raw[df_raw["state"] == state]["area"].unique())
            area  = st.selectbox("City / Area", areas)
            num_stations = st.slider(
                "Number of Monitoring Stations", 1, 40, 2,
                help="How many air quality monitoring stations are in this area."
            )

        with col2:
            st.markdown("<div class='section-title'>🌬️ Pollutant & Time</div>", unsafe_allow_html=True)
            prominent_pollutants = st.selectbox(
                "Prominent Pollutant(s)",
                pollutants,
                index=pollutants.index("PM2.5") if "PM2.5" in pollutants else 0,
                help="The main pollutant(s) measured at this location."
            )
            col_m, col_y = st.columns(2)
            month = col_m.selectbox("Month", list(range(1, 13)), index=0,
                                    format_func=lambda m: [
                                        "Jan","Feb","Mar","Apr","May","Jun",
                                        "Jul","Aug","Sep","Oct","Nov","Dec"
                                    ][m-1])
            year  = col_y.selectbox("Year", [2022, 2023, 2024, 2025], index=2)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Air Quality", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Running SVM prediction..."):
            result = M.predict(
                state=state,
                area=area,
                prominent_pollutants=prominent_pollutants,
                num_stations=num_stations,
                month=month,
                year=year,
            )

        st.markdown("<div class='section-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)
        st.markdown("<div class='result-panel'>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns([3, 1.5, 1.5])
        with r1:
            st.markdown(
                f"<div class='status-badge' style='background:{result['color']};'>"
                f"{result['label']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:#475569; font-size:0.92rem; margin-top:6px;'>"
                f"{result['description']}</p>",
                unsafe_allow_html=True,
            )
        with r2:
            st.metric("SVM Confidence", f"{result['confidence']}%")
        with r3:
            primary = prominent_pollutants.split(",")[0].strip()
            st.metric("Primary Pollutant", primary)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Probability chart ─────────────────────────────────────────────────
        st.markdown(
            "<div class='section-title' style='margin-top:20px;'>📊 Class Probability Breakdown</div>",
            unsafe_allow_html=True,
        )
        proba_df = (
            pd.DataFrame(list(result["all_proba"].items()), columns=["Status", "Probability (%)"])
            .sort_values("Probability (%)", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(9, 3.5))
        bar_colors = [M.STATUS_COLORS.get(s, "#888") for s in proba_df["Status"]]
        bars = ax.barh(
            proba_df["Status"], proba_df["Probability (%)"],
            color=bar_colors, edgecolor="white", height=0.55,
        )
        for bar, val in zip(bars, proba_df["Probability (%)"]):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", fontsize=9, color="#334155",
            )
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.set_title("SVM Output Probabilities per Class", fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Insight note ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='info-card' style='margin-top:14px;'>
        ℹ️ <strong>How the prediction works:</strong> The SVM was trained on
        <em>location, pollutant type, and time</em> — not on the raw AQI numeric value
        (which would directly reveal the answer). This makes the model genuinely learn
        geographic and seasonal patterns in air quality.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown("""
    <div class='page-hero'>
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Understand the structure, distributions, and patterns in India's CPCB air quality dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    df_raw = get_raw_data()

    # ── Snapshot ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Dataset Snapshot</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",  f"{len(df_raw):,}")
    c2.metric("Columns",     df_raw.shape[1])
    c3.metric("States",      df_raw["state"].nunique())
    c4.metric("Cities",      df_raw["area"].nunique())

    with st.expander("Preview first 10 rows"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # ── AQI value + status distribution ──────────────────────────────────────
    st.markdown("<div class='section-title'>AQI Value & Status Distribution</div>", unsafe_allow_html=True)
    fig = M.plot_aqi_distribution(df_raw)
    st.pyplot(fig)
    plt.close()

    # ── Class imbalance callout ───────────────────────────────────────────────
    counts = df_raw["air_quality_status"].value_counts()
    severe_pct = round(counts.get("Severe", 0) / len(df_raw) * 100, 2)
    st.markdown(f"""
    <div class='warn-card'>
    ⚠️ <strong>Class Imbalance:</strong> The dataset is heavily skewed toward
    <em>Satisfactory</em> ({counts.get('Satisfactory',0):,} rows) and
    <em>Moderate</em> ({counts.get('Moderate',0):,} rows), while
    <em>Severe</em> has only {counts.get('Severe',0):,} rows ({severe_pct}% of total).
    The SVM uses <code>class_weight='balanced'</code> to compensate.
    </div>
    """, unsafe_allow_html=True)

    # ── Top pollutants ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Top 10 Prominent Pollutants</div>", unsafe_allow_html=True)
    top_poll = df_raw["prominent_pollutants"].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.barh(top_poll.index[::-1], top_poll.values[::-1], color="#6366f1", height=0.6)
    ax2.set_xlabel("Count", fontsize=10)
    ax2.set_title("Most Common Pollutant Combinations", fontweight="bold", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Average AQI by state ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Average AQI by State — Top 10 Most Polluted</div>", unsafe_allow_html=True)
    state_aqi = (
        df_raw.groupby("state")["aqi_value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    colors3 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.85, len(state_aqi)))
    ax3.barh(state_aqi.index[::-1], state_aqi.values[::-1], color=colors3[::-1], height=0.6)
    ax3.set_xlabel("Average AQI", fontsize=10)
    ax3.set_title("Top 10 Most Polluted States (Mean AQI)", fontweight="bold", fontsize=12)
    ax3.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Pollutant × status heatmap ────────────────────────────────────────────
    st.markdown("<div class='section-title'>Pollutant × Air Quality Status Heatmap</div>", unsafe_allow_html=True)
    primary     = df_raw["prominent_pollutants"].str.split(",").str[0].str.strip()
    top_primary = primary.value_counts().head(8).index
    heat_df     = df_raw.copy()
    heat_df["primary_pollutant"] = primary
    heat_df = heat_df[heat_df["primary_pollutant"].isin(top_primary)]
    pivot   = heat_df.groupby(["primary_pollutant", "air_quality_status"]).size().unstack(fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax4,
        linewidths=0.4, linecolor="#f1f5f9",
    )
    ax4.set_title("Pollutant × Status Count Heatmap", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Air Quality Status", fontsize=10)
    ax4.set_ylabel("Primary Pollutant", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── Monthly trend ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Monthly Average AQI Trend</div>", unsafe_allow_html=True)
    df_dated = df_raw.copy()
    df_dated["date"]  = pd.to_datetime(df_dated["date"], dayfirst=True, errors="coerce")
    df_dated["month"] = df_dated["date"].dt.month
    monthly = df_dated.groupby("month")["aqi_value"].mean().reindex(range(1, 13))

    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax5.fill_between(range(1, 13), monthly.values, alpha=0.18, color="#0ea5e9")
    ax5.plot(range(1, 13), monthly.values, color="#0ea5e9", linewidth=2.5, marker="o", markersize=6)
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(month_labels, fontsize=9)
    ax5.set_ylabel("Average AQI", fontsize=10)
    ax5.set_title("Monthly Average AQI (All States)", fontweight="bold", fontsize=12)
    ax5.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Metrics":
    st.markdown("""
    <div class='page-hero'>
        <h1>🤖 SVM Model Performance</h1>
        <p>Evaluation on the held-out 20% test set — trained without AQI value leakage.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        st.markdown("""
        <div class='warn-card'>
        ⚙️ <strong>Training SVM for the first time…</strong> This takes 2–5 minutes.
        After saving, every future launch is instant.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Loading / training model..."):
        model, le_target, encoders, results = get_model()

    if results is None:
        with st.spinner("Computing evaluation metrics..."):
            results = get_eval_results(model, le_target)

    acc = results["accuracy"]

    # ── Leakage warning ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='info-card'>
    ℹ️ <strong>Data leakage fix applied:</strong> <code>aqi_value</code> and <code>aqi_bucket</code>
    were removed from features. Both columns directly encode the target
    (<em>air_quality_status</em> is just a categorical bin of <em>aqi_value</em>).
    Including them yielded a spurious 100% accuracy. The model now genuinely learns from
    geographic, pollutant, and temporal signals — giving a realistic accuracy of
    <strong>~{acc*100:.1f}%</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level metrics ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Overall Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",     f"{acc*100:.2f}%")
    m2.metric("Model",        "SVM (RBF)")
    m3.metric("C Parameter",  "10")
    m4.metric("Test Split",   "20%")

    # ── Per-class report ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Per-Class Classification Report</div>", unsafe_allow_html=True)
    report = results["report"]
    rows   = []
    for cls in results["classes"]:
        r = report[cls]
        rows.append({
            "Class":     cls,
            "Precision": f"{r['precision']:.3f}",
            "Recall":    f"{r['recall']:.3f}",
            "F1-Score":  f"{r['f1-score']:.3f}",
            "Support":   int(r["support"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    fig_cm = M.plot_confusion_matrix(results["cm"], results["classes"])
    st.pyplot(fig_cm)
    plt.close()

    # ── SVM confidence chart ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>SVM Mean Confidence per Class</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.88rem; margin-bottom:12px;'>"
        "SVM has no feature importances — instead we show the mean prediction confidence "
        "per class on the held-out test set.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Computing confidence chart..."):
        fig_conf = M.plot_svm_class_confidence(model, le_target)
    st.pyplot(fig_conf)
    plt.close()

    # ── Model parameters ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Model Parameters & Rationale</div>", unsafe_allow_html=True)
    params = {
        "Kernel — RBF":               "Maps inputs into higher-dimensional space to find non-linear class boundaries.",
        "C = 10":                     "High regularisation → tighter fit. Balanced against margin via cross-validation.",
        "gamma = 'scale'":            "Auto-sets RBF bandwidth based on feature variance (= 1 / (n_features × X.var())).",
        "class_weight = 'balanced'":  "Upweights rare classes (Severe: 555 rows) to avoid majority-class dominance.",
        "probability = True":         "Platt scaling on top of SVM scores — enables confidence % outputs.",
        "StandardScaler":             "SVM is scale-sensitive; all features normalised to zero-mean, unit-variance.",
        "Leakage guard":              "aqi_value & aqi_bucket excluded — they deterministically encode the target label.",
    }
    for param, exp in params.items():
        st.markdown(
            f"<div class='insight-card' style='margin-bottom:6px;'>"
            f"<strong><code>{param}</code></strong><br>"
            f"<span style='color:#475569; font-size:0.9rem;'>{exp}</span></div>",
            unsafe_allow_html=True,
        )