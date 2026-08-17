# Vehicle Fault Diagnosis and Early Warning System

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

Fleet vehicles today are serviced reactively — a fault is caught after it happens, not before — and one maintenance schedule doesn't fit every type of user. We give a fleet manager an early warning signal instead: for any vehicle in the fleet, **forecast whether a fault will occur in the next 6 hours**, using only its own hourly sensor telemetry, so maintenance can be scheduled proactively rather than after breakdown.

**Data:** All 4 vehicles/usage profiles (rare, moderate, daily, heavy) in the EV Sensors Driving-Pattern Diagnostics dataset, merged: 175,176 hourly readings, 2020–2024, 10 raw sensors per reading (SOC, SOH, Charging Cycles, Battery Temp, Motor RPM, Motor Torque, Motor Temp, Brake Pad Wear, Charging Voltage, Tire Pressure).

**Task framing:** The forecasting problem is converted into a binary classification task — for every hour, look 6 hours ahead in that vehicle's own history and label it fault / no fault (`Fault_Within_6h`). One yes/no decision per hour, not a continuous forecast.

**Evaluation metric:** F2-score (fault as the positive class). Precision and recall combined, recall weighted twice as heavily as precision. Chosen over macro F1 (a `DummyClassifier` that never predicts a fault still scores macro F1 ≈ 0.476 — deceptively OK-looking) and over plain recall (gameable by predicting "fault" every time). F2 can't be gamed by either extreme and matches the real cost asymmetry: missing a fault is worse than a false alarm.

**Validation:** Stratified train/test split (140,067 train / 35,017 test rows), per the explicit instruction for this track. A separate, more leakage-conservative single-vehicle chronological split lives in the legacy notebooks (01–07).

# Installation & Setup

## 1. Requirements

All dependencies are listed in **`requirements.txt`**:

```
numpy · pandas · scikit-learn · xgboost
matplotlib · seaborn · jupyter · streamlit · plotly
```

## 2. Clone the repository

```bash
git clone git@github.com:harshavardhanratnala-hvr/agentic-ai-vehicle-fault-diagnosis-copilot-v2.git
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

Shared conventions — kept identical across all notebooks so the models stay comparable:

- **Same seed:** `RSEED = 42`.
- **Same split:** stratified train/test split on `Fault_Within_6h`.
- **Same metric:** F2-score on the fault class (primary), plus recall, precision, and accuracy for context.
- **Feature engineering runs per vehicle** (grouped by usage profile) so one vehicle's rolling/lag history never leaks into another's.

**To launch the dashboard:**

```bash
streamlit run app.py
```

The dashboard is a 5-page app: Home, Live Diagnosis, Try Your Own, Compare Models, About. A sidebar picker (vehicle profile + scenario) persists across pages.

______________________________________________________________________

# Contents of the Repository

```text
agentic-ai-vehicle-fault-diagnosis-copilot-v2/
├── notebooks/
│   ├── 01_eda.ipynb ... 07_evaluation.ipynb           # legacy: single-vehicle, chronological split
│   ├── 08_eda_all_vehicles.ipynb                      # ACTIVE track (all 4 vehicles) starts here
│   ├── 09_coach_spec_simple_baseline.ipynb            # required baseline: logistic regression
│   ├── 10_feature_engineering_all_vehicles.ipynb
│   ├── 11_feature_selection_all_vehicles.ipynb
│   ├── 12_tree_models_all_vehicles.ipynb              # RandomizedSearchCV tuning
│   ├── 13_expanded_lag_features_all_vehicles.ipynb
│   ├── 14_gridsearch_tuning_all_vehicles.ipynb        # full GridSearchCV -- production model
│   └── 15_gridsearch_lagged_all_vehicles.ipynb
├── docs/
│   ├── Business_Goal_and_Process.md                   # business goal, metric, process, results
│   ├── Classifier_and_Dashboard_FAQ.md                # plain-language FAQ on classifier.py & dashboard
│   ├── Presentation_Notes.md                          # story arc, key numbers, talking points
│   └── Rebuild_Plan.md                                # legacy single-vehicle plan (superseded)
├── data/
│   ├── raw/
│   └── processed/
│       └── scenarios/                                 # 12 demo fixtures (4 vehicles x 3 scenarios)
├── models/
│   ├── xgboost_all_vehicles_gridsearch.joblib         # production model (F2 = 0.706)
│   ├── logistic_regression_all_vehicles.joblib        # baseline model (F2 = 0.328)
│   └── *_results*.json                                # scoreboard numbers per model
├── src/
│   └── classifier.py                                  # classify_fault / classify_fault_baseline
├── app.py                                             # Streamlit entry point / page router
├── dashboard_lib.py                                    # shared dashboard UI (theme, sidebar, cards)
├── pages/
│   ├── 0_Home.py
│   ├── 1_Live_Diagnosis.py
│   ├── 2_Try_Your_Own.py
│   ├── 3_Compare_Models.py
│   └── 4_About.py
├── slides/
├── requirements.txt
└── README.md
```

______________________________________________________________________

# Documentation

- **`Business_Goal_and_Process.md`**
  The formal write-up: business goal, why F2 was chosen as the metric, the baseline model and score, the full notebook 08–15 process (feature engineering, feature selection, tuning, full GridSearchCV), the final scoreboard, and the known tradeoffs (stratified split, merged vehicles). Read this one for exact numbers and reasoning.
- **`Classifier_and_Dashboard_FAQ.md`**
  A plain-language companion FAQ: what `classifier.py` does step by step, where the 0.5 flag threshold came from, why a prediction needs 24 hours of history for a 6-hour-ahead target, the multi-page dashboard structure, and the honest before/after of the GridSearchCV tuning (F2 0.594 → 0.706).

______________________________________________________________________

# Notebook Walkthrough

### `08_eda_all_vehicles.ipynb`

Explores all 4 usage profiles on the merged, raw dataset. Confirms 0 missing values across all 175,176 rows / 4 vehicle timelines. Establishes the headline EDA finding used throughout the rest of the project: fault rate genuinely differs by usage profile, before modeling, not discovered after.

### `09_coach_spec_simple_baseline.ipynb`

Builds the required baseline: **Logistic Regression**, balanced class weights, all 10 raw sensor columns, present-moment values only, no history. Deliberately simple — a baseline, not a competitor to the advanced model.

### `10_feature_engineering_all_vehicles.ipynb`

Computes rolling/lag features (6h/12h/24h windows, 1h/3h/6h lags) **per vehicle**, plus calendar features. This is the feature-engineering logic `src/classifier.py` reproduces at inference time.

### `11_feature_selection_all_vehicles.ipynb`

Ranks candidate features by mutual information against a random-noise benchmark, cross-checked with permutation importance. Run on the full training/test set, no subsampling.

### `12_tree_models_all_vehicles.ipynb`

Trains Random Forest and XGBoost on the selected features, tuned via `RandomizedSearchCV` scored on F2. This is the model that was later superseded by the full grid search in notebook 14.

### `13_expanded_lag_features_all_vehicles.ipynb` → `15_gridsearch_lagged_all_vehicles.ipynb`

Explores an expanded lag-feature set and re-runs `GridSearchCV` on it; kept for reference but not promoted to production — see the FAQ for the honest before/after comparison.

### `14_gridsearch_tuning_all_vehicles.ipynb`

Re-tunes XGBoost with a full `GridSearchCV` (not randomized) on the same 53 selected features. This is where the deployed model (`models/xgboost_all_vehicles_gridsearch.joblib`) comes from — a real, substantial improvement over the notebook 12 model.

______________________________________________________________________

# Results

| Model | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.517 | 0.133 | 0.328 |
| XGBoost (RandomizedSearchCV, notebook 12) | 0.854 | 0.268 | 0.594 |
| **XGBoost (GridSearchCV, full grid — production)** | **0.855** | **0.416** | **0.706** |

Baseline → production model: **F2 up 2.15x** (0.328 → 0.706), recall up from **52% to 85%**, with precision more than tripling versus the randomized-search model (0.268 → 0.416).
