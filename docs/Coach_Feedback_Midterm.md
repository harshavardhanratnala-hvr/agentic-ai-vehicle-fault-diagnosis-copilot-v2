# Midterm coach feedback — binding for the final presentation

Status: received after the midterm, in progress. Everything in the "Action list" section
must be reflected in the final deck. Do not silently drop any of this in two weeks;
if a decision changes, update this file and say why.

## Where things stand (implemented so far)

All 8 items below are implemented in `slides/Bolt-Project-update-3-fixed.pptx`. Deck is now
15 slides, speaker split Ghada 1-5 / Harsha 6-10 / Hüseyin 11-15.

- Slide order (Marinka's flow, adopted): title → problem → stakeholders → **dashboard demo**
  (moved up so the audience sees something real early) → evaluation metric → dataset →
  **new classification-conversion slide** → workflow → EDA → baseline → advanced model →
  rolling/lag explainer → tech stack → RAG next-up → thank you.
- Evaluation metric slide: removed "roughly" (F2's recall weighting is exact). Added the
  classification-framing line bridging business goal to metric.
- New slide, right after the dataset slide: "How we turned this into a yes/no question" —
  explains the forecast-to-classification conversion (group by usage profile, engineer a
  forward-looking 6-hour target) without ever using the term "time series." States the
  assumption (4 usage profiles assumed representative) and the limitation (more
  vehicles/battery types would help) explicitly, in their own callout cards.
- ML workflow slide: stripped to high-level pipeline only (EDA, Feature Engineering,
  Feature Selection, Model & tune), no jargon.
- Rolling/lag explainer slide (after the advanced model slide): concrete examples
  (`Motor_Temp_lag_3h`, `Motor_RPM_roll_mean_24h`), the deployment implication, and the
  generalization limitation.
- RAG/next-up slide: simplified to a 2-column table (Workstream, Status) — dropped the dense
  third "Note" column so it can be talked through verbally with icons/status only, not read
  aloud line by line.
- Title slide: added an amber car icon above the title, on the navy background.
- Problem statement slide: bullets shortened to short phrases ("Caught reactively, not
  proactively" / "No early-warning signal today" / "One schedule doesn't fit every
  vehicle"), font bumped to 22pt bold, and the blank right side now holds a soft warning-
  triangle icon with the caption "A car fails, you find out later. We want to catch it
  earlier." as a speaking cue.
- Tech stack slide: notes already used present tense ("this is the technology we are
  using"), confirmed, no change needed.
- Dashboard demo slide: added a caption under the existing "Probability of Fault" box —
  "e.g. 80% probability, flagged above a 33% threshold" — showing the probability-plus-
  threshold idea directly on the mockup.
- Speaking notes rewritten for all 15 slides to match the new order, speaker split, and
  hand-offs (baked into the pptx's notes pages).

## Still open (process items — not deck edits)

1. **Team practice.** Run through the full deck together at least once before presenting,
   with a timer, and redistribute slide count/time if one person is running long. Current
   split is an even 5/5/5 by slide count, but actual talking time per slide varies (the
   advanced-model and classification-conversion slides run longer) — time it for real.
2. **Warren's point, worth restating live even without full RAG:** being able to write good
   structured/unstructured queries against the NHTSA corpus is a demonstrable skill on its
   own, independent of whether the full agent loop ships.

## Scope decision: RAG vs. improving the model

Coach's explicit ask: spend the remaining time before the final improving the classifier,
not advancing RAG. Diagnosis quality comes first; RAG matters "only then." This does not
mean drop RAG — the NHTSA corpus is already collected — it means don't rush the rest of the
pipeline (chunking, retrieval, agent loop, dashboard) at the cost of the model.

Both coaches independently said: it's fine, and better, to ship one polished, well-understood
component than several half-built ones. Keep RAG as clearly-labeled future work if it isn't
fully built by the final. Warren's point is worth keeping regardless of whether the full RAG
system ships: being able to write good structured/unstructured queries against the NHTSA
corpus is a demonstrable skill on its own, worth mentioning even without the full agent loop.

## Raw feedback (for reference, lightly cleaned up from a recorded transcript)

**Coach (Rakib):**
- Slide 4: don't say "roughly 2x" — F2's recall weighting is a precise mathematical
  definition (beta=2), not approximate.
- Bigger comment: the deck jumps from business goal straight to metric with no line saying
  "this is a classification task." Bridge that gap explicitly, even if another team already
  covered F2 (in which case, briefly credit that and remind the audience anyway).
- Slide 5 (dataset): liked it, but watch font sizes.
- Slide 11 (RAG next-up, old numbering): was skipped entirely when presenting. Either
  present every slide you include, or cut the ones you don't intend to talk to.
- Suggested reordering: business goal → stakeholders → demo (so the audience sees something
  real) → then back to the workflow slide.
- Slide 6 (ML workflow, old numbering): too abstract at this point — "rolling, lag" undefined.
  Keep this slide high-level; explain rolling/lag/calendar features with concrete examples
  later, in the advanced model section. Also explicitly connect this to a real-life scenario:
  in the deployed app, the baseline model only needs current sensor values, but a model using
  lag features needs the last N hours of readings from the user — show this to make the
  classification framing concrete.
- On RAG vs. model: has the team collected RAG documents? (Yes.) Suggestion: spend the
  remaining week improving the model instead — diagnosis quality first, RAG only matters
  after that's solid.
- Demo idea: show the predicted probability (e.g. logistic regression's probability output),
  not just a binary flag, with a threshold and a recommendation once it's crossed.

**Assistant coach (Marinka):**
- Title slide needs a stronger visual hook (e.g. a car image), not just text on navy.
- Problem statement slide: too much unused blank space on the right; left side has long,
  small-font sentences. Use short bullet phrases as speaker reminders, not full sentences —
  reduces cognitive load for the audience. Say the full sentence aloud instead of writing it.
- Suggested rephrase for delivery: "a car fails, you find out about it later; we want to help
  you find out earlier" — ease into the jargon rather than opening with "faults are caught
  reactively."
- Liked the team-intro portion of the title slide.
- ML workflow slide: font size can be larger, there's a lot of unused space.
- Tech stack slide: change future tense ("technology we're going to be using") to present
  tense ("technology we are using"), since the project is already underway.
- RAG next-up table: too dense to read aloud and follow at the same time. Simplify — icons/
  status only, explain detail verbally.
- Time distribution: redistribute slide count/time more evenly across the three speakers;
  practice together to calibrate (the team had not practiced together before this run).
- Model slide was confusing without clarifying whether this is time series or classification,
  what's being forecast, and whether the features are themselves time series. Fix: when
  introducing the dataset, spend a slide explaining how the forecasting problem was
  converted into a classification task (group by usage profile, engineer a forward-looking
  target from historical data). Don't use the term "time series" — just describe the
  conversion directly. State the assumption (4 usage profiles assumed representative of the
  fleet) and the limitation (more vehicles/battery types would make it more reliable).
  Also state plainly that using lag/rolling features means the real-world use case needs a
  window of historical measurements (e.g. the last 6 hours), not just an instant reading.
- It's fine to keep RAG as future work rather than rushing it — better one polished, fully
  understood component than several half-finished ones, especially given the one-month
  timeline.

**Warren (fellow participant, not a coach — noted for context):**
- Even without a fully built RAG system, demonstrating the ability to write good structured/
  unstructured queries and filters against an existing document corpus (here, the NHTSA
  data) is valuable on its own and worth showcasing.

## Team split used for the midterm (kept here for continuity)

Ghada: slides 1-4 (title, problem, target user, evaluation metric)
Harsha: slides 5-10 (dataset, ML workflow, EDA, baseline model, advanced model, rolling/lag
explainer)
Hüseyin: slides 11-14 (tech stack, RAG next-up, dashboard demo, thank you)

This split is a reasonable starting point for the final too, but should flex if the
reordering in item 2 above is adopted (e.g. moving the demo earlier would shift who presents
when).

## Team split for the final (15 slides, reordering adopted)

Ghada: slides 1-5 (title, problem, target user, dashboard demo, evaluation metric)
Harsha: slides 6-10 (dataset, classification-conversion, ML workflow, EDA, baseline model)
Hüseyin: slides 11-15 (advanced model, rolling/lag explainer, tech stack, RAG next-up, thank you)

Practice together with a timer before the final — this split is even by slide count, not
yet verified even by talking time.
