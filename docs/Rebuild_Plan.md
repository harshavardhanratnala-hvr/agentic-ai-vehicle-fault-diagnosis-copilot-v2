# Rebuild plan — Vehicle Fault Forecasting (single vehicle, time series)

> **STATUS: superseded / legacy.** This plan covers the **single-vehicle track only**
> (notebooks 01–07). The project's active line of development is the **all-vehicle track**
> (notebooks 08–12) — see `Business_Goal_and_Process.md` for the current business goal, metric,
> and results. This document is kept for reference (it explains the reasoning behind the
> single-vehicle track, which still exists as a separate, more leakage-conservative body of work),
> not as the current plan.

Status: draft for review before any code is written. Nothing here has been built yet.

---

## 1. Business goal (single sentence, defensible)

For one continuously-monitored vehicle, forecast whether a fault will occur in the next 6 hours, using only its own sensor history, so its operator can schedule preventive maintenance before a breakdown — proven first on one vehicle, explicitly not yet claimed to generalize to other vehicles or usage patterns.

This replaces the old "4 usage profiles merged into one model" framing entirely. No merging, no cross-profile fairness question, no dataset-limitation argument to defend — those problems are designed out of scope, not patched around.

## 2. Explicit non-goals (say these out loud to the coach, don't let them surface as surprises later)

- Not claiming the model works on any vehicle other than the one it's trained on.
- Not doing root-cause diagnosis (that's a different, classification-shaped problem — could be a v2, not this).
- Not building a recommendation model — repair suggestions still come from the LLM + RAG layer at inference time, not from a trained/labeled recommendation dataset (unchanged from before, still correct).

## 3. Data scope

- One vehicle only: `heavy_user`, ~43,777 hourly rows, Jan 2020 – Dec 2024 (5 years).
- Raw sensors: SOC, SOH, Charging_Cycles, Battery_Temp, Motor_RPM, Motor_Torque, Motor_Temp, Brake_Pad_Wear, Charging_Voltage, Tire_Pressure.
- Target: `Fault_Within_6h` (binary, forward-looking 6h window over `is_fault`), same construction logic as before — that part was already correct and brute-force-verified. Reused, not rebuilt from scratch.

## 4. What was actually wrong last time, and the fix for each — this is the part to walk through with him first

| # | Problem (confirmed) | Fix in the rebuild |
|---|---|---|
| 1 | 4 vehicles merged into 1 model; no fair way to test cross-profile generalization | Single vehicle only. Eliminated, not mitigated. |
| 2 | Raw `timestamp` dropped with nothing to replace it | Replaced with `hour_of_day`, `day_of_week`, `month_of_year` (repeating, model can reuse across train/test) + `hours_since_start` (trend/vehicle-age piece). Raw timestamp itself still excluded as a literal feature — that part was already right, just needs the replacement columns added. |
| 3 | Feature-relevance / permutation-importance analysis was run but never used to select features — all 50 features went into every model regardless | Feature selection becomes a required pipeline step, not a plot: rank by MI + permutation importance, decide a cutoff, document it, only the surviving features go into the model. |
| 4 | No naive/persistence baseline computed before real models | Baseline required and run first: e.g. "predict fault iff a fault occurred in the last 6h" and "always predict majority class." Real models must beat these, on paper, before being taken seriously. |
| 5 | No hyperparameter tuning — manual settings only | Add a documented tuning pass (grid or random search) for at least the shipped model, with the search space and chosen values written down. |
| 6 | Trend-trap risk never checked — SOH/Charging_Cycles trend monotonically, chronological split guarantees test-set values are more extreme than anything trees saw in training | Explicit check before modeling: compare train vs. test min/max for every monotonic-trend feature. If test exceeds train's range, document it as a known model limitation (tree models can't extrapolate) rather than discovering it live in a demo. |
| 7 | Single train/test split only, no robustness check across multiple time windows | Add rolling-origin validation (train on an earlier window, test on the next, slide forward, repeat 3-4 times) to confirm the model's performance isn't a fluke of one particular cutoff. |
| 8 | 24/48 NaN rows in the 6h/12h targets — real reason (end-of-timeline), already handled correctly, but undocumented anywhere visible | Keep the fix, add one explicit markdown cell stating why they exist and confirming they're dropped, so it's never mistaken for a bug again. |

## 5. Methodology sequence (in order — nothing skipped, nothing done out of order this time)

1. **EDA on the single vehicle** — line plot of key sensors over the full 5 years, fault-occurrence timeline, class balance of `Fault_Within_6h`, basic seasonality check (does fault rate vary by month or day-of-week).
2. **Feature engineering** — rolling mean/std, lags, deltas on the 4 sequence sensors (unchanged, this part was sound); add the calendar/trend replacement columns from fix #2; explicit trend-trap range check from fix #6.
3. **Feature selection** — MI + permutation importance, pick a defensible cutoff, document which features survive and why.
4. **Baseline models** — persistence/naive rule, majority-class rule. Recorded and never deleted from the final comparison table.
5. **Logistic regression** — balanced class weights, scaled inputs, as the first real model. Coefficients reviewed for sanity (do the signs make physical sense).
6. **Tree models** — Random Forest and XGBoost, using only the selected features from step 3, with a documented hyperparameter search.
7. **Evaluation** — chronological holdout (last ~20% of this one vehicle's timeline) as the primary split, plus rolling-origin validation across 3-4 windows for robustness. Metrics: macro F1, fault-class recall/precision — accuracy explicitly excluded as misleading (documented why, once, so it stops being re-litigated).
8. **(Optional, time permitting) Neural net comparison** — LSTM, same evaluation protocol, only if steps 1-7 are solid and there's time left. Not required to hit a working deliverable.
9. **Write-up** — one page per model, methodology decisions log kept as we go (not reconstructed after the fact under time pressure).

## 6. What stays unchanged from the original plan (don't rebuild what isn't broken)

- RAG over NHTSA recall/complaint/TSB data, filtered to battery/electrical tags — untouched, this was never part of the dispute.
- LLM ticket-generation step — untouched.
- Agent orchestration (classify → search → generate ticket) — untouched, still 2-3 tools.
- Streamlit/FastAPI deployment plan — untouched.

## 7. Repo structure (new repo)

```
notebooks/
  01_eda.ipynb                       # single-vehicle EDA
  02_feature_engineering.ipynb       # rolling/lag + calendar/trend features + trend-trap check
  03_feature_selection.ipynb         # MI + permutation importance, documented cutoff
  04_baselines.ipynb                 # naive + majority-class, required reference point
  05_logistic_regression.ipynb
  06_tree_models.ipynb               # RF + XGBoost, tuned, on selected features only
  07_evaluation.ipynb                # chronological holdout + rolling-origin CV, final scoreboard
docs/
  Capstone_Project_Plan.md           # updated with new scope, business goal, non-goals
  Decisions_Log.md                   # one entry per methodology decision, dated, with reasoning
data/                                 # gitignored except .gitkeep
models/
src/                                  # unchanged: rag/, agent/, api/ scaffolding, populated in later weeks
```

## 8. Acceptance criteria per stage (so "done" isn't vague)

- EDA: at least one plot showing the target's relationship to time (seasonality or lack thereof), stated explicitly.
- Feature engineering: trend-trap check output included in the notebook, not just run and discarded.
- Feature selection: a written cutoff rule (e.g. "top N by MI" or "importance > threshold"), and the resulting feature list saved to a file the modeling notebooks import, not retyped.
- Baselines: appear in the same final comparison table as every other model, every time results are reported.
- Tree models: hyperparameter search space and chosen values written in a markdown cell.
- Evaluation: both the single chronological holdout number and the rolling-origin average reported side by side.

## 9. Open questions to settle before writing any code

1. Confirm with the coach: is the single-vehicle, forecasting-only scope acceptable, or does he still want the multi-vehicle/classification reframe? (This plan assumes the single-vehicle path discussed above.)
2. 12h horizon — keep as a secondary target alongside 6h, or drop to reduce scope? (Recommend: keep 6h only for the rebuild, revisit 12h later if time allows.)
3. Old repo — archive or delete? (Recommend: keep as `-legacy` for reference, don't delete history.)

---

Nothing here has been executed. Once you confirm the business goal, the fix list in section 4, and the open questions in section 9, the next step is building notebook 01 only — one stage at a time, reviewed before moving to the next, not the whole pipeline built in one pass like before.
