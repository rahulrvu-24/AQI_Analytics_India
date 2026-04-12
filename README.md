# 🌫️ AQI Air Quality Classifier

An end-to-end Machine Learning project that predicts **India's Air Quality Status**
(Good → Severe) from real CPCB monitoring data, with a full **Streamlit web app**.

---

## 📁 Project Structure

```
AQI_Project/
├── aqi.csv          ← Raw dataset (India CPCB, ~235K records)
├── model.py         ← Full ML pipeline: load → preprocess → train → predict
├── app.py           ← Streamlit web application (4 pages)
├── notebook.ipynb   ← Step-by-step EDA + training walkthrough
└── README.md        ← You are here
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AQI_Project.git
cd AQI_Project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501** in your browser.

### 5. (Optional) Train the model from CLI
```bash
python model.py
```

### 6. (Optional) Open the notebook
```bash
jupyter notebook notebook.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
jupyter
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit jupyter
```

---

## 🎯 Problem Statement

**Input features:**
- State & City/Area
- Number of AQI monitoring stations
- Prominent pollutant(s) — PM2.5, PM10, O3, CO, SO2, NO2, etc.
- AQI Value (3–500)
- Month & Year

**Predict:** Air Quality Status
| Status       | AQI Range | Health Impact |
|--------------|-----------|---------------|
| Good         | 0–50      | Minimal risk  |
| Satisfactory | 51–100    | Acceptable    |
| Moderate     | 101–200   | Sensitive groups affected |
| Poor         | 201–300   | General public affected   |
| Very Poor    | 301–400   | Health emergency warnings |
| Severe       | 401–500   | Avoid all outdoor activity |

---

## 🧠 ML Pipeline (model.py)

| Step | Function | Description |
|------|----------|-------------|
| 1 | `load_data()` | Read aqi.csv into DataFrame |
| 2 | `preprocess()` | Drop nulls/constants, parse date |
| 3 | `engineer_features()` | Add primary_pollutant, multi_pollutant flag, aqi_bucket |
| 4 | `encode_features()` | Label-encode state, area, pollutants |
| 5 | `train()` | RandomForestClassifier, save model |
| 6 | `evaluate()` | Accuracy, F1, confusion matrix |
| 7 | `predict()` | Inference on single input dict |

---

## 🖥️ Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project intro, AQI category guide, pipeline overview |
| 🔮 Predict | Interactive form — select state, city, AQI, pollutant → get prediction |
| 📊 EDA | Distribution charts, top pollutants, state rankings, heatmaps |
| 🤖 Model Metrics | Accuracy, per-class report, confusion matrix, feature importance |

---

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AQI ML project with Streamlit app"
git remote add origin https://github.com/YOUR_USERNAME/AQI_Project.git
git branch -M main
git push -u origin main
```

---

## 📌 Key Findings

- `aqi_value` is the dominant feature — AQI status is largely deterministic from the raw score.
- `primary_pollutant` differentiates cases at boundary zones between two statuses.
- Winter months (Oct–Jan) show elevated PM2.5 due to crop burning in northern states.
- Delhi, Uttar Pradesh, and Bihar have the highest average AQI in the dataset.
