# Classifier & dashboard — plain-language FAQ

Companion to `docs/Business_Goal_and_Process.md` (which has the formal numbers and tables).
This doc is the "wait, why did we do that again?" reference — written the way it was explained
in conversation, so it's easier to recall than the technical write-up.

---

## What is `src/classifier.py` actually doing?

It's one function, `classify_fault(readings_df)`, that turns raw sensor readings into a risk
call. Three steps happen inside it:

1. **Load the trained model.** `models/xgboost_all_vehicles.joblib` is a saved copy of the
   XGBoost model after training — training happens once in a notebook and takes real time;
   saving it to a `.joblib` file means you never have to retrain it again, you just reload the
   finished model from disk. Think of it like saving a Word doc: the file *is* the trained
   model, ready to use.
2. **Rebuild the features.** The model wasn't trained on raw sensor values directly — it was
   trained on rolling averages, lags, and calendar features computed from them (notebook 10).
   `classify_fault` recomputes those exact same features from whatever raw readings you hand
   it, so the input matches what the model actually learned from.
3. **Predict.** `model.predict_proba(...)` is a built-in method on the loaded model — not
   something you import separately, it comes for free once the model object is loaded. It
   returns the probability of a fault in the next 6 hours. `classify_fault` turns that number
   into a plain "Normal" / "Flagged" call and hands both back.

## Where did the 0.5 "flag" threshold come from?

An earlier sketch (not built by us) suggested flagging above 33%. We checked that against the
actual test data instead of trusting it: F2 (our real metric) peaks at exactly 0.5, not 0.33 —
0.594 at 0.5 versus 0.541 at 0.33. Flagging at 0.33 would also mark 44% of all readings as
risky, which is too noisy for anyone to act on. 0.5 isn't a compromise, either — it's the same
threshold already baked into every recall/precision/F2 number we've quoted everywhere else.
One number, one meaning, no separate "dashboard threshold" to keep in sync.

## Why does one prediction need 24 hours of history if the target is only 6 hours ahead?

These are two different clocks, not one:

- **6 hours** is how far *forward* the model is predicting — "will there be a fault between
  now and 6 hours from now." That's the target label, and it never changes.
- **24 hours** is how far *backward* the model needs to look to compute its own features —
  the biggest rolling window in the feature set (`_roll_mean_24h`, `_roll_std_24h`) needs a
  full day of trailing readings before it's calculable at all.

They sit on opposite sides of "now." A single live prediction needs 24 hours of history
minimum — the 48-hour window used in the demo sketch is just 24 hours of warm-up plus 24 hours
of predictions stacked into a trend line, not a modeling requirement.

## Does the vehicle need to be switched on / driving for this to work?

Not driving, specifically — but it does need to be **connected and reporting hourly**. Modern
EVs already report battery/telematics data through an onboard telematics unit whether parked,
charging, or driving, the same way remote monitoring and over-the-air updates already work. So
this isn't as strict a limitation as "the car must be running."

The honest limitation is what happens when that connection *breaks*: a dead 12V battery, a
connectivity dead zone, a service visit where it's unplugged, or an older fleet vehicle with no
telematics hardware at all. Any gap resets the 24-hour warm-up — no fresh prediction until a
full clean day of data has come back in. Worth stating as a known limitation next to the
"4 usage profiles" one already in `Business_Goal_and_Process.md`, not something to wait for the
coach to ask about.

**Follow-up: does that cost battery power overnight?** Yes, a little — this is the same
"vampire drain" every modern EV already has, where a low-power telematics/BMS module stays
awake on the 12V accessory battery to report status and monitor the traction battery even while
parked, the same way a phone drains slightly just checking for signal with no apps open. Worth
naming as a real, non-zero cost rather than pretending the telemetry is free. It's also smaller
than it could be specifically *because* our model only needs hourly readings, not a continuous
stream — a unit that wakes up once an hour and goes back to sleep draws much less power than
one holding a live connection open, so the hourly design is the power-conscious choice, not just
the modeling-convenient one.

## What's the realistic dashboard plan?

**The idea:** a single-page Streamlit app. Pick a vehicle/profile, pick a scenario, hit run,
see risk status, probability, a trend line, and a short "why" explanation.

**The flow, in order:**
1. User picks a vehicle profile and a scenario from two dropdowns.
2. The app loads that scenario's pre-saved 48-hour reading window — no live upload during the
   demo, so nothing can choke on bad input in front of an audience.
3. `classifier.py` rebuilds the features and calls the model (`classify_fault_batch` for the
   whole window, so there's a trend to show, not just one number).
4. Streamlit renders four things: a risk badge (Normal/Flagged), the probability, a 24-point
   trend chart, and a short SHAP-based explanation of which readings drove that prediction.

**Why a scenario picker instead of live streaming:** live telemetry streaming in real time
during a presentation is the unrealistic version — too much can go wrong live, and it doesn't
actually demonstrate anything a canned example can't. Two or three pre-loaded windows (normal
driving, gradual degradation, pre-fault escalation) picked from a dropdown gets you the same
"look, it reacts to different real situations" moment, live, with zero risk of it breaking on
stage.

**What's realistic to build:** the flow above, using data you already have, calling code you
already have (`src/classifier.py`). Roughly a day of work.

**What's a nice-to-have, not the target:** arbitrary file upload with validation, multiple
vehicles compared side by side, saving predictions to a database, anything resembling real
deployment (hosting, auth, a backend service).

**What to skip entirely:** real-time streaming input, retraining from the UI, the mock
service-scheduling tool. All three are scope creep relative to "show the model working live,"
which is the actual goal.

## A real bug the "why" panel caught, worth knowing about

Building the explanation feature exposed a genuine bug in `classify_fault`, not just a UI
detail: `hours_since_start` (vehicle age) was being computed relative to the *start of
whatever window you passed in*, not the vehicle's actual first day in service. So a vehicle
that had been running for 226 hours looked, to the model, like a 47-hour-old vehicle, every
single time — because each fresh 48-hour window restarted the clock at zero. That's a real
distribution shift the model was never trained to see, and the "why" panel immediately flagged
it by showing `hours_since_start` as a top driver in cases where it shouldn't have mattered
that much.

Fixed by adding a `vehicle_start_time` argument to `classify_fault`, `classify_fault_batch`,
`explain_prediction`, and `engineer_features` — pass in when the vehicle actually went into
service (for this dataset, `2020-01-01` for all four), and the feature computes correctly
regardless of how short a window you're scoring. Good reminder that an explanation panel isn't
just a nice demo feature — it's a real sanity check on the pipeline, and it already paid for
itself once.

## Explanation ≠ RAG — don't conflate these if asked

Two unrelated things share the word "explanation" in this project:

- **RAG's job:** hand back a citation to an actual document — "here's the NHTSA recall record
  that matches this fault code." Text corpus, retrieval, citations.
- **The dashboard's "why" panel:** explain the classifier's own number — which sensor readings
  pushed *this* prediction up, using SHAP (a standard technique for tree models). No documents
  involved at all.

If a coach asks about "the explanation," check which one they mean before answering.
