import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = "aqi.csv"
MODEL_FILE   = "svm_model.pkl"
ENCODER_FILE = "encoders.pkl"

# ── Constants ────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["state", "area", "prominent_pollutants", "primary_pollutant"]

FEATURE_COLS = [
    "state",
    "area",
    "number_of_monitoring_stations",
    "prominent_pollutants",
    "primary_pollutant",
    "multi_pollutant",
    "month",
    "year",
]

TARGET_COL = "air_quality_status"

STATUS_COLORS = {
    "Good":         "#00C853",
    "Satisfactory": "#64DD17",
    "Moderate":     "#FFD600",
    "Poor":         "#FF6D00",
    "Very Poor":    "#DD2C00",
    "Severe":       "#6A1A4C",
}

AQI_CATEGORY_INFO = {
    "Good":         (0,   50,  "Air quality is satisfactory and poses little or no risk."),
    "Satisfactory": (51,  100, "Acceptable; some pollutants may concern very sensitive people."),
    "Moderate":     (101, 200, "Sensitive groups may experience health effects."),
    "Poor":         (201, 300, "Everyone may begin to experience adverse health effects."),
    "Very Poor":    (301, 400, "Health warnings of emergency conditions for everyone."),
    "Severe":       (401, 500, "Serious health effects. Avoid all outdoor activities."),
}


# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# STEP 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["note", "unit"], errors="ignore", inplace=True)
    df.dropna(subset=[TARGET_COL], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["year"]  = df["date"].dt.year.fillna(0).astype(int)
    df.drop(columns=["date"], inplace=True)
    return df


# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["primary_pollutant"] = df["prominent_pollutants"].str.split(",").str[0].str.strip()
    df["multi_pollutant"]   = df["prominent_pollutants"].str.contains(",").astype(int)
    return df


# STEP 4 — ENCODE CATEGORICALS
# ══════════════════════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, ENCODER_FILE)
    else:
        encoders = joblib.load(ENCODER_FILE)
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df


# STEP 5 — TRAIN (SVM)
# ══════════════════════════════════════════════════════════════════════════════
def train(df: pd.DataFrame):
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET_COL])
    X = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

    svm_pipeline.fit(X_train, y_train)
    joblib.dump({"model": svm_pipeline, "le_target": le_target}, MODEL_FILE)
    return svm_pipeline, X_test, y_test, le_target


# STEP 6 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, le_target) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report":   classification_report(
            y_test, y_pred,
            target_names=le_target.classes_,
            output_dict=True,
        ),
        "cm":      confusion_matrix(y_test, y_pred),
        "classes": le_target.classes_,
    }


# STEP 7 — PREDICT (single sample)
# ══════════════════════════════════════════════════════════════════════════════
def predict(
    state: str,
    area: str,
    prominent_pollutants: str,
    num_stations: int = 1,
    month: int = 1,
    year: int = 2024,
) -> dict:
    artifacts = joblib.load(MODEL_FILE)
    model     = artifacts["model"]
    le_target = artifacts["le_target"]
    encoders  = joblib.load(ENCODER_FILE)

    primary_pollutant = prominent_pollutants.split(",")[0].strip()
    multi_pollutant   = int("," in prominent_pollutants)

    row = {
        "state":                         state,
        "area":                          area,
        "number_of_monitoring_stations": num_stations,
        "prominent_pollutants":          prominent_pollutants,
        "primary_pollutant":             primary_pollutant,
        "multi_pollutant":               multi_pollutant,
        "month":                         month,
        "year":                          year,
    }

    df_s = pd.DataFrame([row])

    for col in CATEGORICAL_COLS:
        le = encoders[col]
        df_s[col] = df_s[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    proba = model.predict_proba(df_s[FEATURE_COLS])[0]
    y_enc = int(np.argmax(proba))
    label = le_target.inverse_transform([y_enc])[0]
    conf  = round(float(proba[y_enc]) * 100, 1)

    return {
        "label":       label,
        "confidence":  conf,
        "color":       STATUS_COLORS.get(label, "#888"),
        "description": AQI_CATEGORY_INFO.get(label, ("", "", ""))[2],
        "all_proba":   dict(zip(le_target.classes_, (proba * 100).round(1))),
    }


def aqi_value_to_status(aqi_value: int) -> dict:
    """Deterministic lookup: AQI value → status (no ML, just the official bins)."""
    for status, (lo, hi, desc) in AQI_CATEGORY_INFO.items():
        if lo <= aqi_value <= hi:
            return {
                "label":       status,
                "color":       STATUS_COLORS[status],
                "description": desc,
                "range":       f"{lo}–{hi}",
            }
    return {"label": "Unknown", "color": "#888", "description": "", "range": "—"}


# FULL PIPELINE — run once to build & save the model
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(data_path: str = DATA_PATH) -> dict:
    print("Loading data...")
    df = load_data(data_path)
    print(f"   {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("Preprocessing...")
    df = preprocess(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Encoding categoricals...")
    df = encode_features(df, fit=True)

    print("Training SVM (StandardScaler + SVC · RBF kernel) ...")
    model, X_test, y_test, le_target = train(df)

    print("Evaluating...")
    results = evaluate(model, X_test, y_test, le_target)

    print(f"\nAccuracy : {results['accuracy']*100:.2f}%")
    print(classification_report(
        y_test, model.predict(X_test), target_names=le_target.classes_
    ))
    print(f"\nModel saved    → {MODEL_FILE}")
    print(f"Encoders saved → {ENCODER_FILE}")

    results["model"]         = model
    results["feature_names"] = FEATURE_COLS
    return results


# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=0.5, linecolor="#e0e0e0",
    )
    ax.set_title("Confusion Matrix — SVM Classifier", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    return fig


def plot_svm_class_confidence(model, le_target):
    df = load_data()
    df = preprocess(df)
    df = engineer_features(df)
    df = encode_features(df, fit=False)

    le_t = le_target
    y    = le_t.transform(df[TARGET_COL])
    X    = df[FEATURE_COLS]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    proba     = model.predict_proba(X_test)
    y_pred    = np.argmax(proba, axis=1)
    mean_conf = []

    for i, cls in enumerate(le_t.classes_):
        mask = y_pred == i
        mean_conf.append(proba[mask, i].mean() * 100 if mask.sum() > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [STATUS_COLORS.get(c, "#888") for c in le_t.classes_]
    bars = ax.bar(
        le_t.classes_, mean_conf,
        color=bar_colors, edgecolor="white", width=0.55,
    )
    for bar, val in zip(bars, mean_conf):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Mean Confidence (%)", fontsize=11)
    ax.set_title("SVM Mean Prediction Confidence per Class", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_aqi_distribution(df_raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_raw["aqi_value"], bins=50, color="#4A90D9", edgecolor="white")
    axes[0].set_title("Distribution of AQI Values", fontweight="bold")
    axes[0].set_xlabel("AQI Value")
    axes[0].set_ylabel("Count")
    axes[0].spines[["top", "right"]].set_visible(False)

    counts     = df_raw["air_quality_status"].value_counts()
    bar_colors = [STATUS_COLORS.get(s, "#888") for s in counts.index]
    axes[1].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", width=0.6)
    axes[1].set_title("Records per Air Quality Status", fontweight="bold")
    axes[1].set_xlabel("Status")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    run_pipeline()