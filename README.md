[README.md](https://github.com/user-attachments/files/34223332/README.md)
# Beach Water Quality Analysis — Five-Model Comparison

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/pandas-3.0-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-red)
![Dataset](https://img.shields.io/badge/dataset-data.gov-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An academic predictive-analytics project that asks one simple question:

> **Can ordinary hourly sensor readings tell when Chicago beach water is more turbid (cloudier) than usual?**

Turbidity means how cloudy the water is. Cloudy water can come from waves stirring up sand, storm runoff, or algae. This project uses the public **"Beach Water Quality – Automated Sensors"** dataset from data.gov and compares **five machine-learning methods** to see which one predicts relative turbidity best.

**Important:** this project predicts *relative turbidity level* only. It does **not** predict whether swimming is safe, and it does not use any bacteria or microbial data — the dataset does not contain any.

---

## 📌 Dataset

| Item | Detail |
|---|---|
| Name | Beach Water Quality – Automated Sensors |
| Publisher | Chicago Park District / City of Chicago |
| Catalog page | https://catalog.data.gov/dataset/beach-water-quality-automated-sensors |
| Direct CSV | https://data.cityofchicago.org/api/views/qmqz-2xku/rows.csv?accessType=DOWNLOAD |
| Snapshot used | Downloaded 31 Oct 2026 — 49,151 rows × 10 columns |
| After cleaning | 45,444 readings (2013-08-30 → 2026-09-25) |
| Sensors report | Water temperature, turbidity, wave height, wave period, battery voltage, transducer depth, every hour |

Beaches covered include Ohio Street, Calumet, Montrose, Osterman, 63rd Street and Rainbow.

---

## 🎯 Objectives

1. **Clean the raw sensor file** — handle the `-100000` error readings, missing values, duplicate rows and impossible negative values.
2. **Explore the data (EDA)** — find patterns in turbidity across beaches, water temperature, month and hour of day.
3. **Define a workable target** — label each reading as *Higher* or *Lower* turbidity using the median of the **training** data (1.25 NTU), because no official safe/unsafe label exists.
4. **Build five models with the same simple code structure** and compare them fairly on the same test records.
5. **Find the best-fit model** and report honestly whether it is actually better than a dumb guess.

---

## 🧹 Preprocessing and feature engineering

- Replace `-100000` sensor error codes with missing values.
- Parse the timestamp column into real dates and times.
- Drop rows missing timestamp, water temperature, turbidity, wave height or wave period.
- Drop exact duplicate rows and rows with negative turbidity or wave values.
- Keep rows with missing `Transducer Depth` (it is mostly empty and not used as an input).
- **Feature engineering:** extract `Month` and `Hour` from the timestamp; scale all inputs with `StandardScaler` fitted on training data only.
- **Model inputs (5):** Water Temperature, Wave Height, Wave Period, Month, Hour.
- **Turbidity is never an input** — it is the answer being predicted.

### Train / test split
Sensor readings are time-dependent, so a random split would leak the future.
The **older 80% of timestamps train**, the **newer 20% test** (test starts 2019-06-03 19:00).
36,355 training rows, 9,089 test rows.

---

## 🤖 The five models

Each model sits in its own short, readable block: create → fit → predict → print metrics.

| # | Model | Type | Settings |
|---|---|---|---|
| 1 | Linear Regression | Numeric baseline | default |
| 2 | Logistic Regression | Classification | default |
| 3 | Decision Tree | Classification | `max_depth=8, random_state=42` |
| 4 | K-Nearest Neighbors | Classification | `n_neighbors=5` |
| 5 | K-Means Clustering | Unsupervised | `n_clusters=2, n_init=10, random_state=42` |

Linear Regression predicts the 0/1 label as a number and is thresholded at 0.5 only for comparison. K-Means is unsupervised: its two clusters are named *Lower* / *Higher* using **training labels only**, then scored on the test set as a reference comparison.

---

## 📊 Results

Higher turbidity is the positive class.

| Model | Accuracy | Precision | Recall | Higher F1 |
|---|---:|---:|---:|---:|
| Linear Regression | 0.5747 | 0.7154 | 0.4176 | 0.5274 |
| **Logistic Regression** | 0.5778 | 0.7213 | 0.4190 | 0.5301 |
| **Decision Tree** | **0.5781** | 0.6864 | 0.4742 | 0.5609 |
| KNN | 0.5622 | 0.6552 | 0.4846 | 0.5572 |
| K-Means (mapped) | 0.5281 | 0.5691 | 0.6984 | **0.6271** |

**Best-fit model: Decision Tree — highest accuracy at 57.81%**, with Logistic Regression almost identical at 57.78%. K-Means gives the highest Higher-class F1 (0.6271) among the five.

### The honest part
A model that **always** answers "Higher" scores 56.83% accuracy and a **0.7247** Higher F1 — better F1 than every fitted model. So the real conclusion is:

> Water temperature, waves, month and hour **do not explain turbidity well enough** for strong prediction. Decision Tree wins this comparison, but none of the five is a reliable turbidity predictor on its own.

Extra outputs: Linear Regression MAE 0.4514, MSE 0.2306, RMSE 0.4802, R² 0.0601. K-Means test silhouette 0.2577, adjusted Rand index ≈ 0.0002 (near-zero agreement with the turbidity categories).

---

## 📁 Repository files

```
Beach_Project/
├── README.md                                     ← this file
├── Beach_water_quality_project.py                ← full analysis code
├── Beach_Water_Quality_Automated_Sensors.csv     ← dataset snapshot (49,151 rows)
├── Beach_Model_Comparison.csv                    ← saved results table
├── Beach_Analysis.md                             ← detailed notes, limits, reproduction
├── Beach_Analysis_Output.txt                     ← printed console output of the run
├── IEEE_Paper_Beach_Water_Quality.pdf / .docx    ← 6-page IEEE-format report
└── IEEE_Paper_Beach_Water_Quality_v2.pdf / .docx ← report using the project's own charts
```

Charts produced by the script: turbidity box plot, readings-by-beach bar chart, temperature-by-turbidity histogram, correlation heatmap, accuracy and F1 comparison bars, one confusion matrix per model, and a K-Means cluster scatter.

---

## ▶️ How to run

```bash
git clone <your-repo-url>
cd Beach_Project
pip install numpy pandas matplotlib seaborn scikit-learn
python Beach_water_quality_project.py
```

Keep the `.py` file and the CSV in the same folder. The script prints the analysis, shows the charts, and saves `Beach_Model_Comparison.csv` plus the accuracy, F1 and cluster images.

Tested on Python 3.13.12 with numpy 2.4.3, pandas 3.0.3 and scikit-learn 1.9.1.

---

## ⚠️ Limitations

- Single chronological hold-out; no cross-validation and no confidence intervals.
- No microbial or safety labels exist in this dataset, so "Higher/Lower turbidity" is **not** "unsafe/safe".
- Beach name is only used for an exploratory chart — pooling different beaches without location features is a weakness.
- Sensor calibration changes, seasonal gaps and drifting readings over a 13-year span limit how far these numbers generalise.
- The downloaded CSV is kept so the run is reproducible; a fresh live download can shift dates, thresholds and scores.

## 🔜 Possible next steps

Add a training-fitted beach encoding, time-aware cross-validation, lagged features (previous hour's readings), and documented quality-control rules from the sensor publisher.

---

## 📄 License and credit

Data is open and published by the **Chicago Park District / City of Chicago** through data.gov. Code in this repository is for academic coursework.

*Author: Biswajith Tharigopula — School of Computer Science and Engineering, Lovely Professional University.*
