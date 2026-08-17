# Vehicle Fault Diagnosis & Repair Copilot

**Team NodePair** · Ghada · Harsha · Hüseyin

Capstone Project — neue fische Data Science & AI Bootcamp

Dataset: [Kaggle — EV Sensors Driving-Pattern Diagnostics 2020 to 24](https://www.kaggle.com) (CC BY 4.0)

______________________________________________________________________

# Overview & Navigation

- [Project Summary](#project-summary)
- [Installation & Setup](#installation--setup)
- [How to Run / Reproduce](#how-to-run--reproduce)
- [Contents of the Repository](#contents-of-the-repository)
- [Documentation](#documentation)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Results](#results)

______________________________________________________________________

# Project Summary

Fleet vehicles today are serviced reactively, a fault is caught after it happens, not before, and one maintenance schedule doesn;t fit every type of users.
we want to give a fleet manager an early warning signal instead: for any vehicle in the fleet, **forecast whether a fault will occur in the next 6 hours**, using only its own hourly sensor telemetry, so maintenance can be scheduled proactivaly rather than after breakdown.

**Data:** All 4 vehicles/usage profiles(daily, heavy, moderate , rare) in the EV senesor driving-pattern diagnostics dataset, merged 175,176 hourly readings, 2020–2024 10 raw sensor per reading(SOC, SOH, Charging Cycles, Battery Temp, Motor RPM, Motor Torque, Motor Temp, Brake Pad Wear, Charging Voltage, Tire Pressure).

**Task framing:** The forecasting problem is converted into a binary classification task, for every hour, look 6 hours ahead in that vehicule's own history and label it fault / no fault (`Fault_Within_6h`). One yes /no decision per hour, not a continuous forecast.

**Evaluation metric:** F2-score (fault as the positive class)
precision and recall combined, recall weignted exactly 2x as heavely as precision. Chosen over macro F1(a `DummyClassifier` that never predictes a fault atil scores macro F1 ≈ 0.476 -> deceptively
OK-looking) and over plain recall(gameable by predicting "fault"every time).
F2 can't be gamed by either extreme and matches the real cost asymmetry: Missing a fault is worse than a false alarm.

**Validation:** Stratified train/test split( 140,067 train / 35,017 test rows),per the explicit instruction for this track. A separate more leakage-conservative single-vehicle, chronological-split.

# Installation & Setup

## 1. Requirements

All dependencies are listed in **`requirements.txt`**:

```
numpy · pandas · scikit-learn · xgboost
matplotlib · seaborn · jupyter · streamlit
```

## 2. Clone the repository

```bash
git clone < ... >
```

## 3. Set up the environment

**macOS / Git-Bash**

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

______________________________________________________________________

# How to Run / Reproduce

Shared conventions — please keep these identical across all notebooks so the models stay comparable:

- **Same seed:** `RSEED = 42`.
- **Same split:** stratified train/test split on `Fault_Within_6h`.
- **Same metric:** F2-score on the fault class (primary), plus recall precision, and accuracy for context.
- **Feature engineering runs per vehicle** (grouped by usage profile) so one vehicle's rolling/lag history never leaks into another's.

**To launch the dashboard:**

```bash
streamlit run app.py
```

______________________________________________________________________

# Contents of the Repository

```text
vehicle-fault-diagnosis-copilot/
├── notebooks/
│   ├── 01_eda.ipynb ... 07_evaluation.ipynb     # legacy: single-vehicle, chronological split
│   ├── 08_eda.ipynb                             # ACTIVE track (all 4 vehicles)
│   ├── 09_baseline_logreg.ipynb
│   ├── 10_feature_engineering.ipynb
│   ├── 11_feature_selection.ipynb
│   └── 12_tree_models.ipynb
├── docs/
│   ├── Business_Goal_and_Process.md             # business goal, metric, process, results
│   ├── Classifier_and_Dashboard_FAQ.md          # plain-language FAQ on classifier.py & dashboard
│   ├── Presentation_Notes.md                    # story arc, key numbers, talking points
│   └── Rebuild_Plan.md                          # legacy single-vehicle plan (superseded)
├── data/
│   ├── raw/                                                                
│   └── processed/   
├── models/
│   ├── xgboost_all_vehicles.joblib
│   ├── random_forest_all_vehicles.joblib
│   └── baseline_results_all_vehicles.json
├── src/
│   ├── classifier.py                            # classify_fault / classify_fault_batch
├──  app.py                                   # Streamlit demo
├── slides/
│   └── Bolt-Project-update-3-fixed.pptx 
├── requirements.txt
└── README.md
```

______________________________________________________________________

# Documentation

- **`Business_Goal_and_Process.md`**
  The formal write-up: business goal, why F2 was chosen as the metric, the baseline model and score, the full notebook 08–12 process (feature engineering, feature selection, tuning), the final scoreboard, the note on removing subsampling from feature selection and hyperparameter search, the `classify_fault` inference function, RAG corpus status, and the known tradeoffs (stratified split, merged vehicles). Read this one for exact numbers and reasoning.
- **`Classifier_and_Dashboard_FAQ.md`**
  A plain-language companion FAQ: what `classifier.py` does step by step, where the 0.5 flag threshold came from, why a prediction needs 24 hours of history for a 6-hour-ahead target, whether the vehicle needs to be driving (just connected and reporting hourly), the realistic Streamlit dashboard plan, a real bug the "why" panel caught (`hours_since_start` computed from the wrong origin), and the distinction between the dashboard's SHAP-based "why" panel and RAG's document citations. Read this one if you just want the "wait, why did we do that" answer without digging through the formal doc.

______________________________________________________________________

# Notebook Walkthrough

### `08_eda.ipynb`

Explores all 4 usage profiles on the merged, raw dataset. Confirms 0 missing values across all 175,176 rows / 4 vehicle timelines. Establishes the headline EDA finding used throughout the rest of the project: fault rate genuinely differs by usage profile 0.99% (`rare_user`) to 2.57% (`heavy_user`), a 2.6x spread visible *before* modeling, not discovered after.

### `09_baseline_logreg.ipynb`

Builds the required baseline: **Logistic Regression**, balanced class weights, all 10 raw sensor columns, `Charging_Voltage` one-hot encoded rather than scaled (it's categorical, not continuous), stratified train/test split. Target is `Fault_Within_6h`, not the instant fault label (the instant label is close to circular, derived from thresholds the sensors directly trigger). Reports accuracy, recall, precision, and F2 on train and test.

### `10_feature_engineering.ipynb`

Computes rolling/lag features (6h/12h/24h windows, 1h/3h/6h lags) **per vehicle**, to avoid cross-vehicle bleed, plus calendar features (hour-of-day, day-of-week, month-of-year, hours-since-start). Includes the seasonality check that justified the calendar features: hour-of-day shows a 3.7 percentage-point issue-rate spread; month-of-year only 0.4pp. This is the feature-engineering logic `src/classifier.py` reproduces at inference time.

### `11_feature_selection.ipynb`

Ranks 54 candidate features by mutual information against a random-noise benchmark, cross-checked with permutation importance from a reference Random Forest — a feature is kept if *either* signal supports it. Run on the full training/test set (no subsampling, as of 2026-08-11 per coach feedback). Narrows 54 candidates to 53 — only `Motor_Temp_roll_mean_6h` fails both checks. Plots an MI-ranking bar chart (all 54 vs. the noise floor) and a permutation-importance bar chart (MI survivors only).

### `12_tree_models.ipynb`

Trains **Random Forest** and **XGBoost** on the 53 selected features, hyperparameters tuned via `RandomizedSearchCV` scored directly on F2, stratified cross-validation matching the split strategy, search run on the full 140,067-row training set (no subsampling). Plots final feature-importance bar charts for both trained models. This is where the deployed model (`models/xgboost_all_vehicles.joblib`) comes from.

______________________________________________________________________

# Results

| Model | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.517 | 0.133 | 0.328 |
| Random Forest (tuned, 53 selected features) | 0.750 | 0.196 | 0.479 |
| XGBoost (tuned, 53 selected features) | 0.854 | 0.268 | 0.594 |

Baseline → tuned models: **F2 up 1.81x** (0.328 → 0.594), recall up from **52% to 85%**.
