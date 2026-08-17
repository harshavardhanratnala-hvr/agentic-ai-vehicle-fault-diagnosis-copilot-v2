"""Single entry point for the fault classifier: raw sensor readings in, a risk call out.

Wraps `models/xgboost_all_vehicles_gridsearch.joblib` (full-grid `GridSearchCV` tuning on the
full 140,067-row training set, same 53 selected features, F2 = 0.706 — see notebook 14 and
`docs/Business_Goal_and_Process.md`) so the dashboard (or anything else) can call one function
instead of re-running notebook cells. This replaced the earlier `RandomizedSearchCV` model
(F2 = 0.594, notebook 12) once the exhaustive grid search came back with a real, substantial
improvement -- see `docs/Classifier_and_Dashboard_FAQ.md` for the honest before/after.

Usage:
    from classifier import classify_fault
    result = classify_fault(readings_df)
    # {"probability": 0.62, "risk_level": "High", "prediction": True, ...}

`readings_df` must be raw hourly sensor readings for a *single vehicle*, sorted or not
(the function sorts by timestamp itself), with at least 24 consecutive hourly rows ending
at the timestamp you want a prediction for — the model's features include 24-hour rolling
windows and 6-hour lags, so anything shorter can't produce a complete feature row. Required
columns: `timestamp` plus the 10 raw sensor columns listed in RAW_SENSOR_COLS below.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_all_vehicles_gridsearch.joblib"
SELECTED_FEATURES_PATH = BASE_DIR / "data" / "processed" / "selected_feature_cols_all_vehicles.json"

# The baseline model (notebook 09, coach spec): logistic regression on the 10 raw sensor
# columns only, present-moment values, no rolling/lag/calendar engineering at all. This is
# what the dashboard's "Load present readings" path calls -- deliberately simple, on
# purpose, per the coach: "the purpose of a baseline model is not that it does a good job,
# but to see what can be done easily." It needs exactly one row of current readings, no
# history, unlike the advanced model below.
BASELINE_MODEL_PATH = BASE_DIR / "models" / "logistic_regression_all_vehicles.joblib"

# Must match notebooks 08-12 exactly -- these define what "engineered features" means.
RAW_SENSOR_COLS = ["SOC", "SOH", "Charging_Cycles", "Battery_Temp", "Motor_RPM", "Motor_Torque",
                    "Motor_Temp", "Brake_Pad_Wear", "Charging_Voltage", "Tire_Pressure"]
SEQ_SENSORS = ["Motor_RPM", "Motor_Torque", "Motor_Temp", "Battery_Temp"]
ROLL_WINDOWS = [6, 12, 24]
LAGS = [1, 3, 6]
CALENDAR_FEATURE_COLS = ["hour_of_day", "day_of_week", "month_of_year", "hours_since_start"]

# FLAG_THRESHOLD = 0.5, not an arbitrary pick: it's the probability cutoff that maximizes F2
# on the held-out test set for this exact model (checked directly -- F2 peaks at 0.594 right
# at 0.5, the same number already reported everywhere else as this model's F2 score, and drops
# on either side: 0.541 at 0.33, 0.586 at 0.6). It also matches predict()'s default behavior,
# so "flagged" here means exactly what "recall 85.4% / precision 26.8%" already means elsewhere.
# At 0.33 the model would flag 44% of all readings -- too noisy for a fleet manager to act on.
FLAG_THRESHOLD = 0.5

_model = None
_selected_features = None
_baseline_model = None


def _load_model():
    """Load the trained pipeline and selected feature list once, cache for reuse."""
    global _model, _selected_features
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run notebooks 08-12 and then notebook 14 "
                f"(GridSearchCV tuning) first to train and save the model."
            )
        _model = joblib.load(MODEL_PATH)
        with open(SELECTED_FEATURES_PATH) as f:
            _selected_features = json.load(f)
    return _model, _selected_features


def _load_baseline_model():
    """Load the baseline logistic regression pipeline once, cache for reuse."""
    global _baseline_model
    if _baseline_model is None:
        if not BASELINE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{BASELINE_MODEL_PATH} not found. Run notebook 09 first to train and save it."
            )
        _baseline_model = joblib.load(BASELINE_MODEL_PATH)
    return _baseline_model


def engineer_features(readings_df: pd.DataFrame, vehicle_start_time=None) -> pd.DataFrame:
    """Reproduce notebook 10's rolling/lag/calendar feature engineering for one vehicle.

    `vehicle_start_time` should be when *this vehicle* first went into service -- not the start
    of whatever slice of readings you happen to be passing in. `hours_since_start` is trained as
    "vehicle age," and a vehicle that's been running for weeks does not look like a brand-new
    one just because you're only scoring its most recent 48 hours. Omitting this argument falls
    back to treating the first row of `readings_df` as hour zero, which is only correct if
    you're really handing in a vehicle's entire history from day one -- for any shorter window
    (the normal case) that understates the vehicle's real age and can bias this feature.

    Returns a dataframe with all candidate feature columns added; rows without a full
    24h rolling window / 6h lag (the "cold start") will have NaNs in those columns, same
    as in training -- classify_fault drops those before predicting.
    """
    missing = [c for c in ["timestamp"] + RAW_SENSOR_COLS if c not in readings_df.columns]
    if missing:
        raise ValueError(f"readings_df is missing required columns: {missing}")

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in SEQ_SENSORS:
        for w in ROLL_WINDOWS:
            df[f"{col}_roll_mean_{w}h"] = df[col].rolling(window=w, min_periods=w).mean()
            df[f"{col}_roll_std_{w}h"] = df[col].rolling(window=w, min_periods=w).std()
        for lag in LAGS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
        df[f"{col}_delta_6h"] = df[col] - df[col].shift(6)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month_of_year"] = df["timestamp"].dt.month
    start = pd.to_datetime(vehicle_start_time) if vehicle_start_time is not None else df["timestamp"].min()
    df["hours_since_start"] = (df["timestamp"] - start).dt.total_seconds() / 3600

    return df


def _risk_level(probability: float) -> str:
    # Kept to two levels on purpose -- see FLAG_THRESHOLD comment above for why 0.5 and not a
    # borrowed number. A third "Medium" band is easy to add later (e.g. 0.3-0.5 as "Watch") if
    # the team wants it, but it doesn't come from anything in the data, so it's not the default.
    return "Flagged" if probability >= FLAG_THRESHOLD else "Normal"


def classify_fault(readings_df: pd.DataFrame, vehicle_start_time=None) -> dict:
    """Predict fault risk for a single vehicle from its raw hourly sensor readings.

    Uses the most recent row that has a complete feature window (needs >= 24 consecutive
    hourly readings ending at that timestamp). Raises ValueError if no row qualifies.

    Pass `vehicle_start_time` (when this vehicle first went into service) if `readings_df`
    is a recent slice rather than the vehicle's full history -- see engineer_features'
    docstring for why this matters for the "hours_since_start" feature.

    Returns:
        {
            "probability": float,      # model's predicted P(fault within next 6h)
            "risk_level": str,         # "Normal" / "Flagged", see FLAG_THRESHOLD above
            "prediction": bool,        # probability >= FLAG_THRESHOLD
            "timestamp": Timestamp,    # which row the prediction is for
            "n_features_used": int,
        }
    """
    model, selected_features = _load_model()

    engineered = engineer_features(readings_df, vehicle_start_time=vehicle_start_time)
    ready = engineered.dropna(subset=selected_features)
    if ready.empty:
        raise ValueError(
            "No row has a complete feature window -- need at least 24 consecutive hourly "
            "readings (the longest rolling window) ending at the prediction timestamp."
        )

    latest = ready.iloc[[-1]]
    probability = float(model.predict_proba(latest[selected_features])[0, 1])

    return {
        "probability": round(probability, 6),
        "risk_level": _risk_level(probability),
        "prediction": probability >= FLAG_THRESHOLD,
        "timestamp": latest["timestamp"].iloc[0],
        "n_features_used": len(selected_features),
    }


def classify_fault_batch(readings_df: pd.DataFrame, vehicle_start_time=None) -> pd.DataFrame:
    """Same as classify_fault, but returns a row per timestamp that has a complete feature
    window instead of just the latest one -- useful for a dashboard that wants to plot risk
    over time, not just the current reading. See classify_fault for `vehicle_start_time`."""
    model, selected_features = _load_model()

    engineered = engineer_features(readings_df, vehicle_start_time=vehicle_start_time)
    ready = engineered.dropna(subset=selected_features).reset_index(drop=True)
    if ready.empty:
        raise ValueError(
            "No row has a complete feature window -- need at least 24 consecutive hourly "
            "readings (the longest rolling window)."
        )

    probabilities = model.predict_proba(ready[selected_features])[:, 1]
    out = pd.DataFrame({
        "timestamp": ready["timestamp"],
        "probability": probabilities,
    })
    out["risk_level"] = out["probability"].apply(_risk_level)
    out["prediction"] = out["probability"] >= FLAG_THRESHOLD
    return out


def classify_fault_baseline(readings_df: pd.DataFrame) -> dict:
    """Predict fault risk from the single most recent row of *present* sensor readings only --
    no history, no feature engineering, no `vehicle_start_time`. This is the coach-spec
    baseline (notebook 09): logistic regression on the 10 raw sensor columns as-is.

    Deliberately kept dumb -- it's a baseline, not a competitor to the advanced model. It
    exists so the dashboard has a real, separately-clickable "present readings" path, distinct
    from the advanced model's "last 24h" path, instead of only ever showing the advanced
    model's output (coach feedback: the two should be visibly different demo paths, not just
    a metrics table row).

    `readings_df` needs `timestamp` plus the 10 raw sensor columns (RAW_SENSOR_COLS); only the
    latest row by timestamp is used. Uses the model's default 0.5 decision threshold --
    F2-tuning a baseline threshold isn't worth doing since the point of a baseline is that it's
    the simplest reasonable thing, not a tuned competitor to the advanced model.

    Returns the same shape as classify_fault: {"probability", "risk_level", "prediction",
    "timestamp", "n_features_used"}.
    """
    missing = [c for c in ["timestamp"] + RAW_SENSOR_COLS if c not in readings_df.columns]
    if missing:
        raise ValueError(f"readings_df is missing required columns: {missing}")

    model = _load_baseline_model()
    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.sort_values("timestamp").iloc[[-1]]

    probability = float(model.predict_proba(latest[RAW_SENSOR_COLS])[0, 1])
    return {
        "probability": round(probability, 6),
        "risk_level": _risk_level(probability),
        "prediction": probability >= FLAG_THRESHOLD,
        "timestamp": latest["timestamp"].iloc[0],
        "n_features_used": len(RAW_SENSOR_COLS),
    }


def explain_prediction(readings_df: pd.DataFrame, top_n: int = 3, vehicle_start_time=None) -> list:
    """Explain the most recent prediction: which features pushed the probability up or down.

    Uses XGBoost's own built-in SHAP contributions (`pred_contribs=True` on the booster) rather
    than the separate `shap` library -- the `shap` package's TreeExplainer has a known version
    mismatch against recent XGBoost releases (it can't parse the newer `base_score` format), and
    XGBoost computes the exact same SHAP values natively without that dependency. This is not RAG
    -- it explains the classifier's own number from the input features, not a retrieved document.
    See docs/Classifier_and_Dashboard_FAQ.md for that distinction.

    Returns a list of up to `top_n` dicts, sorted by |impact| descending:
        [{"feature": "Motor_Torque_roll_mean_24h", "value": 812.4, "impact": 0.14}, ...]
    A positive "impact" pushed the probability up (more fault-like); negative pushed it down.
    """
    model, selected_features = _load_model()
    engineered = engineer_features(readings_df, vehicle_start_time=vehicle_start_time)
    ready = engineered.dropna(subset=selected_features)
    if ready.empty:
        raise ValueError(
            "No row has a complete feature window -- need at least 24 consecutive hourly "
            "readings (the longest rolling window) ending at the prediction timestamp."
        )
    latest = ready.iloc[[-1]]

    # The pipeline is [scaler, xgb_classifier]; contributions are computed on the scaled inputs
    # the classifier actually sees.
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    scaled = scaler.transform(latest[selected_features])

    booster = clf.get_booster()
    dmatrix = xgb.DMatrix(scaled, feature_names=selected_features)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # last column is the bias term
    feature_contribs = contribs[:-1]

    ranked = sorted(
        zip(selected_features, latest[selected_features].iloc[0].tolist(), feature_contribs),
        key=lambda t: abs(t[2]),
        reverse=True,
    )[:top_n]

    return [
        {"feature": feat, "value": round(float(val), 4), "impact": round(float(impact), 4)}
        for feat, val, impact in ranked
    ]


if __name__ == "__main__":
    # Quick smoke test. data/raw/*.csv has no timestamp column (it's added during cleaning),
    # so use the already-cleaned all-vehicle table and pass only raw sensor columns through,
    # exactly what a real caller would have: readings + a timestamp, nothing engineered yet.
    sample_path = BASE_DIR / "data" / "processed" / "all_vehicles_cleaned.csv"
    if sample_path.exists():
        cleaned = pd.read_csv(sample_path, parse_dates=["timestamp"])
        one_vehicle = cleaned[cleaned["user_profile"] == "heavy_user"].head(48)
        raw_cols_only = one_vehicle[["timestamp"] + RAW_SENSOR_COLS]

        result = classify_fault(raw_cols_only)
        print("classify_fault (first 48h of heavy_user, raw columns only):")
        print(result)

        batch = classify_fault_batch(raw_cols_only)
        print(f"\nclassify_fault_batch: {len(batch)} rows with a complete feature window")
        print(batch.head())
    else:
        print(f"No sample data at {sample_path} -- skipping smoke test.")
