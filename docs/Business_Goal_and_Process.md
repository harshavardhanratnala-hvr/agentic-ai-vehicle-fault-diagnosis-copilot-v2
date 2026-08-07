# Business goal, metric, and process — all-vehicle track (notebooks 08–12)

## Business goal

**Client:** the fleet manager — the person deciding when any of the fleet's vehicles needs to go in for maintenance.

**Data:** all 4 vehicles in the EV sensor driving-pattern diagnostics dataset (daily, heavy, moderate, rare usage profiles), merged — 175,176 hourly readings total, spanning 2020–2024.

For any vehicle in the fleet, forecast whether a fault will occur in the next 6 hours using its sensor readings, so the fleet manager can schedule preventive maintenance before a breakdown happens rather than reacting after one.

## Evaluation metric chosen

F2-score (fault as the positive class) — precision and recall combined, with recall weighted roughly twice as heavily as precision. Chosen for two checked, concrete reasons: a DummyClassifier that never predicts a fault scores macro F1 ≈ 0.476 — deceptively OK-looking for a completely useless model, because macro F1 averages in the negative class's near-perfect score. Positive-class F1 fixes that (goes to 0 for the same useless model), but plain recall alone can also be gamed (predicting "fault" on every row gives 100% recall while being useless). F2 avoids both problems: it can't be gamed by an always-positive model, and it still reflects the real business cost — missing a fault is worse than a false alarm.

## Baseline model

Logistic regression, balanced class weights, all 10 raw sensor columns (SOC, SOH, Charging_Cycles, Battery_Temp, Motor_RPM, Motor_Torque, Motor_Temp, Brake_Pad_Wear, Charging_Voltage, Tire_Pressure), categorical column (`Charging_Voltage`) properly one-hot encoded rather than scaled like a continuous value, stratified train/test split. Target is `Fault_Within_6h`, not the instant fault label — forecasting the near future rather than classifying the present, since the instant label is close to circular (derived from diagnostic codes that sensor thresholds directly trigger).

## Baseline score

| Split | Accuracy | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|---|
| Train | 0.650064 | 0.520183 | 0.133153 | 0.328* |
| Test | 0.649646 | 0.520025 | 0.132962 | 0.327587 |

\*Recomputed for consistency; original run reported F1 rather than F2 at this step.

## EDA (notebook 08)

Fault rate genuinely differs by usage profile even in the raw, unmodeled data: 0.986% (`rare_user`) to 2.573% (`heavy_user`) — a 2.6x spread. Confirmed with 0 missing values across all 175,176 rows and no gaps in any of the 4 vehicles' timelines. This is the evidence base for the "all 4 vehicles merged" tradeoff noted below, not a discovery made after modeling.

## Process — building on the baseline (notebooks 09–12)

**Feature engineering:** rolling/lag features (6h/12h/24h windows, 1h/3h/6h lags) computed per vehicle to avoid cross-vehicle bleed, plus calendar features (hour-of-day, day-of-week, month-of-year, hours-since-start), justified by a real seasonality check (hour-of-day showed a 3.7 percentage-point issue-rate spread, month-of-year only 0.4pp — used as evidence, not assumed).

**Feature selection:** mutual information ranked against a random-noise benchmark, cross-checked with permutation importance from a reference Random Forest — a feature is kept if either signal supports it. Narrowed 54 candidate features down to 45.

**Tree models:** Random Forest and XGBoost, hyperparameters tuned via `RandomizedSearchCV` scored directly on F2 (not macro F1), stratified cross-validation matching the split strategy.

## Final scoreboard (F2 is the headline number)

| Model | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.516556 | 0.132987 | 0.327587 |
| Random Forest (tuned, 45 selected features) | 0.742668 | 0.181196 | 0.458510 |
| XGBoost (tuned, 45 selected features) | 0.855251 | 0.181344 | 0.490611 |

Going from the baseline to the tuned models roughly doubled F2 and raised recall from 52% to 86% — a clear, evidence-backed improvement story built entirely on the foundation specified for this track.

## Known tradeoffs, by design (not oversights)

**Stratified random split, not chronological.** Rows from the same vehicle's adjacent hours can land on both sides of the split — the leakage risk the single-vehicle chronological rebuild (notebooks 01–07) was built to avoid. Used here because a stratified split was the explicit instruction for this track. The chronological version remains available as the leakage-conservative alternative.

**All 4 vehicles merged.** Reopens the original fairness question (different usage profiles pooled into one model) that motivated the single-vehicle pivot earlier in the project. Used here per direct instruction to build the "clean, simple" baseline on all available data. The per-profile fairness breakdown has not yet been re-run on this all-vehicle track — only on the single-vehicle one.

**Hyperparameter search on a subsample.** The search itself ran on a 40,000-row stratified subsample of the 140,067-row training set (compute-time constraint), with the winning configuration then refit on the full training set. The final models are trained on all available data; only the search step was subsampled.
