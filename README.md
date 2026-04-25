# India AQI Analytics Dashboard

An interactive, multi-page analytical dashboard built with **Streamlit** to explore India's air quality data from the Central Pollution Control Board (CPCB).

---

## Overview

This project analyses **235,785 daily AQI readings** collected across **32 states** and **291 cities** in India from **April 2022 to April 2025**. The goal is to surface meaningful patterns in air quality — seasonal trends, geographic hotspots, dominant pollutants, and year-over-year changes — through clean, interactive visualisations.

> No machine learning. Pure data analysis.

---

## Project Structure

```
AQI_Project/
├── app.py                  ← Page 1: Overview (entry point)
├── utils.py                ← Shared: CSS, constants, load_data(), helpers
├── pages/
│   ├── trends.py         ← Page 2: Trends
│   ├── geographic.py     ← Page 3: Geographic
│   ├── pollutants.py     ← Page 4: Pollutants
│   └── explorer.py       ← Page 5: Explorer
├── aqi.csv
├── .gitignore
└── README.md
```

---

## Dataset

| Field | Details |
|---|---|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Period** | April 2022 – April 2025 |
| **Records** | 235,785 daily readings |
| **Coverage** | 32 states · 291 cities |
| **Key columns** | `date`, `state`, `area`, `aqi_value`, `air_quality_status`, `prominent_pollutants`, `number_of_monitoring_stations` |

**AQI Status Categories:**

| Status | AQI Range |
|---|---|
| Good | 0 – 50 |
| Satisfactory | 51 – 100 |
| Moderate | 101 – 200 |
| Poor | 201 – 300 |
| Very Poor | 301 – 400 |
| Severe | 401 – 500 |

---

## Dashboard Pages

### Overview
- National KPIs — total readings, states, cities, mean AQI, % good/satisfactory days
- Air quality status distribution (donut chart + breakdown table)
- National monthly AQI trend (Apr 2022 → Apr 2025)
- Key findings: most/least polluted state, dominant pollutant

### Trends
- Year-over-year monthly AQI comparison (2022 vs 2023 vs 2024)
- Seasonal analysis — average AQI and status mix by season
- Month × Year AQI heatmap
- Status share stacked area chart over time
- Annual summary table (mean AQI, worst month, % poor days)

### Geographic
- Full state AQI ranking with national average reference line
- Top 10 most polluted vs cleanest cities
- State deep-dive — monthly profile, city rankings, status breakdown

### Pollutants
- Primary pollutant frequency chart
- Mean AQI per pollutant type
- Pollutant × Status heatmap
- Pollutant prevalence by season
- State × Pollutant share heatmap

### Explorer
- Interactive multi-select filters (state / year / pollutant)
- Filtered KPIs, AQI histogram, status breakdown
- State comparison bar chart (multi-state selection)
- Monthly AQI trend for any filtered slice
- Raw data preview table (up to 500 rows)

---

## Setup & Run

**Requirements**
```
python >= 3.9
streamlit
pandas
numpy
matplotlib
seaborn
```

**Install dependencies**
```bash
pip install streamlit pandas numpy matplotlib seaborn
```

**Run the app**
```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Key Findings

- **PM10** is the dominant pollutant — present in ~47% of all readings — driven by road dust, construction, and desert geography in Rajasthan, UP, and Haryana.
- **Winter (Oct–Jan)** consistently records the worst air quality nationwide, driven by crop stubble burning, temperature inversions, and reduced wind speeds.
- **Monsoon (Jun–Sep)** delivers the cleanest air — rainfall washes out particulate matter.
- **Satisfactory** is the most common status (37.7%), but **Poor + Very Poor + Severe** together account for ~11.6% of all readings.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Interactive web dashboard |
| Pandas | Data wrangling & aggregation |
| Matplotlib | Charts and visualisations |
| Seaborn | Heatmaps |

---

## Notes

- `.pkl` model files (`rf_model.pkl`, `svm_model.pkl`, `encoders.pkl`) from the earlier ML version of this project have been removed. The project is now analytics-only.
- `__pycache__/` and `*.pkl` are excluded via `.gitignore`.
