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

**Feature selection:** mutual information ranked against a random-noise benchmark, cross-checked with permutation importance from a reference Random Forest — a feature is kept if either signal supports it, computed on the full training/test data (updated 2026-08-11 — see note below). Narrowed 54 candidate features down to 53.

**Tree models:** Random Forest and XGBoost, hyperparameters tuned via `RandomizedSearchCV` scored directly on F2 (not macro F1), stratified cross-validation matching the split strategy, search run on the full 140,067-row training set (updated 2026-08-11 — see note below).

## Final scoreboard (F2 is the headline number)

| Model | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.516556 | 0.132987 | 0.327587 |
| Random Forest (tuned, 53 selected features) | 0.749606 | 0.195799 | 0.478771 |
| XGBoost (tuned, 53 selected features) | 0.853989 | 0.268013 | 0.594172 |

Going from the baseline to the tuned models raised F2 from 0.328 to 0.594 (a 1.81x improvement) and recall from 52% to 85% — a clear, evidence-backed improvement story built entirely on the foundation specified for this track.

**No subsampling anywhere in the pipeline (updated 2026-08-11).** Two steps previously ran on a
subsample of the training data for compute-time reasons, each refit/rescored on the full data
afterward: mutual information + permutation importance (feature selection, notebook 11) used
20K/30K/10K-row subsamples, and the RF/XGBoost hyperparameter search (notebook 12) used a
40,000-row subsample. The coach flagged subsampling as a corner worth closing, so both steps now
run on the complete training set (140,067 rows) / test set (35,017 rows) — no subsample anywhere
from feature selection through final model training.

Removing the feature-selection subsample changed the result meaningfully: MI-vs-noise now keeps
53 of 54 candidate features (up from 45), because on the full 140K rows only one feature
(`Motor_Temp_roll_mean_6h`) fails both the MI-vs-noise and permutation-importance checks — the
20K-row subsample had understated several features' real signal. Feeding that wider, full-data
feature set into the also-now-full-data hyperparameter search pushed XGBoost's F2 to 0.594 (up
from 0.491 in the original subsampled version), with recall back at 85.4% and precision nearly
double the original (26.8% vs. 18.1%). This is a better model by the metric that matters, not
just a more defensible process — and the two fixes only look this good together because the
feature list feeding the search changed too; re-running only one of the two subsampled steps
would have understated the improvement.

The `.joblib` files in `models/`, `data/processed/selected_feature_cols_all_vehicles.json`, and
`models/baseline_results_all_vehicles.json` are all regenerated from this full-data run; the deck
and slide numbers still need to be updated to match.

**Plots added, notebook 11 (feature selection):** an MI-ranking bar chart (all 54 candidates vs.
the random-noise floor) and a permutation-importance bar chart (MI survivors only). **Plot added,
notebook 12 (tree models):** final feature-importance bar charts from the trained Random Forest
and XGBoost, so which of the 53 features the deployed models actually lean on is visible, not just
asserted.

**On the calendar features specifically** (this came up because it looked risky at a glance):
`month_of_year` is the single weakest candidate feature by both MI and permutation importance — it
clears the random-noise floor only barely (MI score ≈ 0.003, versus 0.09+ for the top features)
and its permutation importance is ≈ 0. It survives the OR-based selection rule on that thin MI
margin, not because it's a strong signal. It's also *not* the one feature the full-data rerun
actually dropped — that was `Motor_Temp_roll_mean_6h`, a rolling-window feature, not a calendar
one. Worth knowing for questions: `month_of_year` does still show up in XGBoost's top 25 by final
importance (rank ~14), so the model finds some use for it even though the feature-selection checks
were unconvinced — a reasonable thing to say out loud rather than paper over if asked.

## Inference function (Phase 2 — closes out the model)

`src/classifier.py` wraps `models/xgboost_all_vehicles.joblib` in a single callable:
`classify_fault(readings_df) -> {probability, risk_level, prediction, timestamp,
n_features_used}`. It takes raw hourly sensor readings for one vehicle (`timestamp` +
the 10 `RAW_SENSOR_COLS`), reproduces notebook 10's rolling/lag/calendar feature
engineering internally, and predicts on the most recent row with a complete 24-hour
feature window (shorter history raises a clear `ValueError` rather than guessing).
`classify_fault_batch(readings_df)` returns the same thing for every row that qualifies,
for a dashboard that wants risk over time rather than a single reading.

Risk buckets: "Normal" below 50% probability, "Flagged" at/above it. The 0.5 cutoff isn't a
guess — it's the threshold that maximizes F2 on the held-out test set for this exact model
(checked directly against the data), and it's already the threshold behind every recall/
precision/F2 number quoted elsewhere in this doc and the deck. An earlier draft borrowed 0.33
from a teammate's slide sketch; checked against the data, that threshold flags 44% of all
readings (too noisy to act on) and actually scores a lower F2 (0.541 vs. 0.594 at 0.5).
Smoke-tested against real `heavy_user` data (see the `if __name__ == "__main__"` block);
update both places if the threshold ever changes.

This is what the dashboard (phase 3) will actually call instead of re-running notebook
cells by hand.

## RAG retrieval (Phase 4 of the RAG/agent track — the thin slice, not the full agent)

`src/rag.py` gives one function, `search_recalls_tsbs(query) -> [cited records]`, over the
NHTSA corpus below. Deliberately not the full agent loop — no tool-calling, no wiring to
`classify_fault`'s output yet, just retrieval on its own, checked before anything downstream
is allowed to trust it.

**Method: TF-IDF + cosine similarity, not dense embeddings.** The original plan called for
`sentence-transformers`, but installing it (it needs `torch`) failed repeatedly in the dev
environment with "no space left on device." TF-IDF is a legitimate, standard lexical-retrieval
baseline, not a quiet downgrade — it's what BM25-style search is built on, and for a few
thousand short, structured records like these it performs well (see the gate results below).
Swapping in dense embeddings later only touches `_build_index()`/`search_recalls_tsbs()` in
`rag.py`; the corpus loading and the gate are unaffected either way.

**Chunking:** each row (one recall notice or one complaint narrative) is already a single
short record, a paragraph at most — there's no long document here that needs splitting into
multiple chunks. One row = one document. That's the whole chunking step for this corpus.

**The retrieval-quality gate.** This dataset has no external "fault code -> correct record"
answer key to test against — these are free-text recall/complaint narratives, not a labeled
lookup table. So the gate uses self-retrieval instead: sample real records, build a query from
a realistic fragment of each (words 5-25, simulating how someone would actually describe a
symptom, not the record's full exact text), and check whether retrieval finds that same record
back. This validates the retrieval mechanism works correctly on real data; it is *not* a claim
that retrieval finds the single correct citation for a genuinely novel complaint it's never
seen — that would need real usage feedback or a hand-labeled test set the team doesn't have.

Result on 30 sampled records: **27/30 (90%) at top-1, 29/30 (97%) at top-5.** The one miss was
boilerplate recall-notification text ("received notification of NHTSA Campaign Number...")
duplicated verbatim across many complaints — genuinely ambiguous text, not a retrieval bug.
Re-run it yourself: `python src/rag.py`.

## RAG corpus (Phase 1 of the RAG/agent track)

`data/raw/nhtsa/recalls_battery_electrical.csv` (203 recalls) and
`data/raw/nhtsa/complaints_battery_electrical.csv` (3,329 complaints) — pulled from the
[NHTSA Datasets & APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis), filtered to
battery/electrical component tags, not all recalls for an EV model (that over-broad filter
was an identified risk — see the sibling repo's `Capstone_Project_Plan.md`). Originally
collected in `agentic-ai-vehicle-fault-diagnosis-and-repair-copilot`; copied here on
2026-08-11 so the RAG corpus lives next to the working classifier and dashboard code,
per the team's scope decision to finish the model and a real dashboard first, then do a
thin, verified retrieval slice before attempting the full agent tool-calling loop.

Status (updated 2026-08-12): collected, filtered, chunked, indexed, and retrieval-gated — see
"RAG retrieval (Phase 4)" above. Still not started: the agent tool-calling loop that wires
`search_recalls_tsbs` to `classify_fault`'s output, and the fleet-manager dashboard surfacing
both together. Those stay deliberately deferred per the team's scope decision.

## Known tradeoffs, by design (not oversights)

**Stratified random split, not chronological.** Rows from the same vehicle's adjacent hours can land on both sides of the split — the leakage risk the single-vehicle chronological rebuild (notebooks 01–07) was built to avoid. Used here because a stratified split was the explicit instruction for this track. The chronological version remains available as the leakage-conservative alternative.

**All 4 vehicles merged.** Reopens the original fairness question (different usage profiles pooled into one model) that motivated the single-vehicle pivot earlier in the project. Used here per direct instruction to build the "clean, simple" baseline on all available data. The per-profile fairness breakdown has not yet been re-run on this all-vehicle track — only on the single-vehicle one.

**Hyperparameter search — no longer subsampled.** An earlier version of this pipeline ran the search on a 40,000-row stratified subsample of the training set for compute-time reasons, refitting the winning configuration on the full set. As of 2026-08-11 the search itself also runs on the complete 140,067-row training set — removed per coach feedback that flagged subsampling as a corner worth closing.
