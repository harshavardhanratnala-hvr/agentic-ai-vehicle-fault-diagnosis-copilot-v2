"""The missing link between classify_fault() and search_recalls_tsbs(): generate_ticket().

This is the piece that makes "Repair Copilot" literal rather than aspirational. Before this
file, the project had two working but disconnected halves: a classifier that says *whether*
and *why* a fault is likely, and a retriever that finds *precedent* for a given description.
Nothing turned those into a single written recommendation a fleet manager could act on.

Ticket composition method: a deterministic Python template, not an LLM call. This was a
deliberate choice (see docs/Business_Goal_and_Process.md, "Repair ticket generation (Phase
4b)"): the original plan sketched an LLM writing the final ticket, but that needs a paid API
key with no free tier, which isn't worth the cost or the dependency for a capstone demo. A
template gets the same structured, cited output -- deterministic, free, no network call, no
key to manage -- at the cost of reading a little more like a form than a paragraph. Swapping
in a real LLM call later only means replacing _compose_ticket() below; everything upstream
(query building, retrieval) is unchanged either way.

Flow, per the plan's own diagram:
    classify_fault()        -> risk_level, probability, top driving sensors (explain_prediction)
    -> auto-built query     -> plain-English description of what the sensors are doing
    -> search_recalls_tsbs()-> cited NHTSA recall/complaint precedent
    -> generate_ticket()    -> composes a short, structured, cited ticket from a template

Usage:
    from agent import generate_ticket
    ticket = generate_ticket(readings_df, vehicle_id="heavy_user", vehicle_start_time="2020-01-01")
"""

import re
from pathlib import Path

import pandas as pd

from classifier import classify_fault, explain_prediction
from rag import search_recalls_tsbs

# Turn engineered feature names into plain English for the RAG query and the prompt.
# e.g. "Battery_Temp_roll_mean_6h" -> "battery temperature"; "Motor_Torque_delta_6h" -> "motor torque"
_SENSOR_LABELS = {
    "SOC": "state of charge",
    "SOH": "battery state of health",
    "Charging_Cycles": "charging cycle count",
    "Battery_Temp": "battery temperature",
    "Motor_RPM": "motor RPM",
    "Motor_Torque": "motor torque",
    "Motor_Temp": "motor temperature",
    "Brake_Pad_Wear": "brake pad wear",
    "Charging_Voltage": "charging voltage",
    "Tire_Pressure": "tire pressure",
}
_SUFFIX_RE = re.compile(
    r"_(roll_mean_\d+h|roll_std_\d+h|lag_\d+h|delta_6h)$"
)


def _feature_to_topic(feature_name: str) -> str:
    """'Battery_Temp_roll_mean_6h' -> 'battery temperature'. Falls back to a cleaned-up
    version of the raw name if it's not one of the known sensor columns (e.g. a calendar
    feature slipping into the top factors, which shouldn't drive the query)."""
    base = _SUFFIX_RE.sub("", feature_name)
    return _SENSOR_LABELS.get(base, base.replace("_", " ").lower())


def _build_query(factors: list) -> str:
    """Build a short retrieval query from the classifier's top fault-driving factors.
    Only uses factors that pushed the probability *up* (impact > 0) -- a factor that pushed
    it down isn't part of "what's going wrong," and including it would just add noise to the
    retrieval query."""
    topics = [_feature_to_topic(f["feature"]) for f in factors if f["impact"] > 0]
    if not topics:
        # Nothing pushed risk up -- fall back to all factors so we still retrieve *something*
        # relevant, rather than returning no query at all.
        topics = [_feature_to_topic(f["feature"]) for f in factors]
    # De-dup while preserving order (roll/lag/delta variants of the same sensor collapse to
    # one topic, e.g. three Battery_Temp_* features shouldn't repeat "battery temperature" 3x).
    seen = []
    for t in topics:
        if t not in seen:
            seen.append(t)
    return " ".join(seen)


def _summarize_factors(factors: list) -> str:
    """'battery temperature and motor RPM' -- the plain-English topics driving risk up,
    joined for a readable sentence. Falls back to whatever topics exist if none push risk up."""
    topics = [_feature_to_topic(f["feature"]) for f in factors if f["impact"] > 0] or \
        [_feature_to_topic(f["feature"]) for f in factors]
    seen = []
    for t in topics:
        if t not in seen:
            seen.append(t)
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + f" and {seen[-1]}"


def _compose_ticket(classification: dict, factors: list, citations: list, query: str) -> str:
    """Fill a fixed template from the classification + factors + citations -- no LLM call.
    Deterministic and free; see the module docstring for why this replaced an LLM prompt."""
    topic_summary = _summarize_factors(factors)

    factor_lines = []
    for f in factors:
        direction = "up" if f["impact"] > 0 else "down"
        factor_lines.append(
            f"  - {f['feature']} = {f['value']} (pushed risk {direction}, impact {f['impact']:+.4f})"
        )

    if citations:
        relevant = [c for c in citations if c["score"] >= 0.05]
        evidence_lines = []
        for c in citations:
            relevance = "" if c["score"] >= 0.05 else " (low lexical overlap -- weak match, verify before relying on it)"
            evidence_lines.append(
                f"  - [{c['source'].upper()} {c['citation']}] {c['make']} {c['model']} {c['model_year']} "
                f"(similarity {c['score']:.2f}){relevance}: {c['text'][:200].strip()}..."
            )
        evidence_block = "\n".join(evidence_lines)
        if relevant:
            cause_note = (
                f"Precedent for issues involving {topic_summary} was found in NHTSA records "
                f"(see citation {relevant[0]['citation']} below); the exact cause for this vehicle "
                f"has not been confirmed and requires inspection."
            )
        else:
            cause_note = (
                f"No closely matching NHTSA precedent was found for {topic_summary} "
                f"(best match below has low lexical overlap) -- the driving sensors are still the "
                f"lead to investigate, just without a strong matching historical record."
            )
    else:
        evidence_block = "  (no relevant precedent retrieved)"
        cause_note = (
            f"No matching NHTSA precedent was retrieved for {topic_summary} -- "
            f"treat the sensor readings below as the primary lead."
        )

    return (
        f"SUMMARY: Model flagged a {classification['probability']:.0%} probability of a fault "
        f"within the next 6 hours, driven primarily by {topic_summary}.\n\n"
        f"LIKELY CAUSE: {cause_note}\n\n"
        f"RECOMMENDED ACTION: Inspect the components tied to {topic_summary} at the next available "
        f"service window; prioritize sooner if the vehicle is in active use, since these readings "
        f"are trending in a fault-like direction right now.\n\n"
        f"SUPPORTING EVIDENCE (retrieved for query: {query!r}):\n{evidence_block}\n\n"
        f"DRIVING SENSOR READINGS:\n" + "\n".join(factor_lines) + "\n\n"
        f"CONFIDENCE NOTE: This is a model prediction plus retrieved precedent, not a confirmed "
        f"diagnosis -- a technician should verify before repair."
    )


def generate_ticket(
    readings_df: pd.DataFrame,
    vehicle_id: str = None,
    vehicle_start_time=None,
    top_n_factors: int = 3,
    top_k_citations: int = 3,
) -> dict:
    """Full pipeline: classify -> explain -> retrieve -> compose the ticket from a template.

    Returns:
        {
            "vehicle_id": str | None,
            "risk_level": str, "probability": float,
            "factors": [...],               # from explain_prediction
            "citations": [...],             # from search_recalls_tsbs
            "query_used": str,              # what was searched for, for transparency
            "ticket_text": str | None,      # the composed ticket, None if not flagged
        }

    If the vehicle isn't flagged (risk_level == "Normal"), no retrieval or composition
    happens -- there's nothing to write a repair ticket about. `ticket_text` is None in
    that case; check `risk_level` first.
    """
    classification = classify_fault(readings_df, vehicle_start_time=vehicle_start_time)

    result = {
        "vehicle_id": vehicle_id,
        "risk_level": classification["risk_level"],
        "probability": classification["probability"],
        "timestamp": classification["timestamp"],
        "factors": [],
        "citations": [],
        "query_used": None,
        "ticket_text": None,
    }

    if classification["risk_level"] != "Flagged":
        return result

    factors = explain_prediction(readings_df, top_n=top_n_factors, vehicle_start_time=vehicle_start_time)
    query = _build_query(factors)
    citations = search_recalls_tsbs(query, top_k=top_k_citations)

    result["factors"] = factors
    result["citations"] = citations
    result["query_used"] = query

    result["ticket_text"] = _compose_ticket(classification, factors, citations, query)
    return result


if __name__ == "__main__":
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent
    sample_path = BASE_DIR / "data" / "processed" / "scenarios" / "prefault.csv"
    if not sample_path.exists():
        print(f"No sample data at {sample_path} -- skipping smoke test.")
        sys.exit(0)

    readings = pd.read_csv(sample_path, parse_dates=["timestamp"])
    ticket = generate_ticket(readings, vehicle_id="prefault_scenario", vehicle_start_time="2020-01-01")

    print(f"risk_level: {ticket['risk_level']}  probability: {ticket['probability']}")
    if ticket["ticket_text"]:
        print(f"\nquery_used: {ticket['query_used']!r}")
        print(f"\n--- TICKET ---\n{ticket['ticket_text']}")
    else:
        print("(not flagged -- no ticket generated)")
