# Presentation notes — key facts, decisions, and story for midterm/final

Everything here is real, checked, and reproducible from the notebooks in this repo. Kept as one
place so nothing gets lost between now and presentation day.

---

## The story arc (this is presentation-worthy on its own)

1. Started with 4 vehicles merged into one model, no clear evaluation rigor.
2. Coach challenged: is it fair to pool 4 different usage profiles into one model? → confirmed the
   critique was valid (real, measured recall gap of up to 0.874 vs 0.168 across profiles on the
   original model).
3. Pivoted to a single-vehicle (`heavy_user`), chronological-split, leakage-conservative rebuild
   (notebooks 01–07) to resolve that fairness question directly.
4. Coach then asked for something simpler: all 4 vehicles, raw features only, stratified split,
   logistic regression, as an explicit new baseline.
5. Built that baseline (notebook 08→09), then layered feature engineering, feature selection, and
   tuned tree models on top of it (notebooks 09–12), per direct instruction — this is now the
   **active track**.
6. Along the way, caught and fixed several of our own methodology gaps mid-stream: a categorical
   column being scaled instead of encoded, macro F1 hiding total model failure, plain recall being
   gameable.

**Why this arc matters for the presentation:** it shows iterative, evidence-based development —
every pivot was in response to a specific, checkable critique, not an arbitrary change. That's a
stronger story than "we built it once and it worked."

---

## Business goal (current, active version)

**Client:** the fleet manager — decides when a vehicle needs maintenance.

**Goal:** for any vehicle in the fleet, forecast whether a fault will occur in the next 6 hours
using its sensor readings, so maintenance can be scheduled before a breakdown rather than after one.

**Data:** all 4 usage profiles (daily, heavy, moderate, rare), merged — 175,176 hourly readings,
2020–2024.

---

## How the target was built (good to have ready if asked "what are you actually predicting")

Raw data only has a `DTC` (diagnostic trouble code) column. `Fault_Label` (Normal/Warning/Fault) is
derived from `DTC` via a documented rule (motor/battery codes or multiple simultaneous codes =
Fault; single advisory code = Warning; no code = Normal) — **this is a judgment call the team made,
not something the dataset provides.** `Fault_Within_6h` (the actual model target) is a forward-
looking window: True if a fault/warning occurs anywhere in the next 6 hours. Verified against a
brute-force check, not just trusted.

---

## Evaluation metric journey (own this story, it shows real rigor)

1. Started with macro F1.
2. Coach: "check what DummyClassifier scores." Checked it directly — a model that **never predicts
   a single fault** still scores macro F1 ≈ 0.476. Confirmed, not disputed.
3. This proved macro F1 can look deceptively OK for a useless model, because it averages in the
   negative class's near-perfect score.
4. Switched to positive-class (fault) F1 — but recall alone can also be gamed (predict "fault"
   every time → 100% recall, useless model).
5. Landed on **F2-score** (fault = positive class, recall weighted ~2x precision) — can't be gamed,
   and matches the real business cost asymmetry (missing a fault costs more than a false alarm).

**One-line version for the room:** "We didn't just pick a metric — we checked it against a dummy
model, found a real flaw, and fixed it before it became a problem in front of you."

---

## Final scoreboard (active track, notebooks 08–12)

| Model | Recall (fault) | Precision (fault) | F2 (fault) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.517 | 0.133 | 0.328 |
| Random Forest (tuned, 53 selected features) | 0.750 | 0.196 | 0.479 |
| XGBoost (tuned, 53 selected features) | 0.854 | 0.268 | 0.594 |

Baseline → tuned models: F2 up 1.81x, recall went from 52% to 85%. (Updated 2026-08-11 — feature
selection and hyperparameter search now both run on the full training set, no subsampling. See
`docs/Business_Goal_and_Process.md` for the before/after and why removing the subsample changed
the selected feature count from 45 to 53.)

EDA finding (notebook 08): fault rate genuinely differs by usage profile even in raw data —
0.986% (`rare_user`) to 2.573% (`heavy_user`), a 2.6x spread. Visible before modeling, not
discovered after.

---

## Known limitations — say these plainly, don't wait to be asked

1. **Stratified split, not chronological.** Rows from the same vehicle's adjacent hours can land on
   both sides of the split. Used because it was the explicit instruction for this track. The
   single-vehicle chronological version (notebooks 01–07) avoids this, kept as the leakage-
   conservative alternative.
2. **All 4 vehicles merged** reopens the original fairness question. Per direct instruction; the
   per-profile fairness breakdown has not been re-run on this track yet (only on the single-vehicle
   one).
3. **Hyperparameter search ran on a 40K-row subsample** of the 140K training rows (compute-time
   constraint), final models refit on the full training set.
4. **Logistic regression coefficients ran backwards from physical intuition** on some features
   (e.g. higher `SOH` associated with higher fault probability) in earlier single-vehicle testing —
   traced to `Fault_Label` being derived from `DTC` codes, not from the sensor values themselves.
   A dataset-realism limitation, not a bug.
5. **No historical repair-recommendation data exists.** The dataset only has sensor readings +
   fault labels. The repair-recommendation step (separate from this classifier) will be generated
   via LLM + RAG over NHTSA data at inference time, not learned from labeled examples.

---

## Where everything lives (repo: `agentic-ai-vehicle-fault-diagnosis-copilot-v2`)

- `docs/Business_Goal_and_Process.md` — the current summary (read this first).
- `docs/Presentation_Notes.md` — this file.
- `docs/Rebuild_Plan.md` — legacy, single-vehicle track only (marked superseded at the top).
- `notebooks/08`–`12` — **active track**: EDA → baseline → feature engineering → feature
  selection → tuned tree models, all on all 4 vehicles.
- `notebooks/01`–`07` — separate single-vehicle, chronological-split track, kept for reference.

---

## Still not started (say this honestly if asked about overall project status)

RAG over NHTSA recall/complaint/TSB data, the LLM ticket-generation step, and the agent
orchestration layer are **all still untouched** — everything above is the classifier component
only. Worth flagging proactively in a status update, not waiting to be asked.
