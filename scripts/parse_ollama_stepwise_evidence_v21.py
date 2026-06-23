import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IN_PATH = RESULTS / "ollama_stepwise_evidence_raw.jsonl"
OUT_PATH = RESULTS / "ollama_stepwise_evidence.csv"


def clean_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = candidate.replace("True", "true").replace("False", "false")
    candidate = candidate.replace("None", "null")

    # Repair common unquoted-key pattern:
    # {supports_forward: true, confidence: 0.95}
    candidate = re.sub(r'([{,]\s*)(supports_forward|confidence)\s*:', r'\1"\2":', candidate)

    return candidate


def find_json_candidates(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Prefer fenced JSON blocks.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for item in fenced:
        yield item

    # Then any object-looking span.
    objects = re.findall(r"\{[^{}]*supports_forward[^{}]*\}", text, flags=re.IGNORECASE | re.DOTALL)
    for item in objects:
        yield item

    # Finally whole text.
    if text.startswith("{") and text.endswith("}"):
        yield text


def extract_json_object(text: str):
    for candidate in find_json_candidates(text):
        fixed = clean_candidate(candidate)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict) and "supports_forward" in obj:
            return obj

    return None


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    s = str(value).strip().lower()
    if s in ("true", "yes", "1", "forward", "event_a_before_event_b", "a_before_b"):
        return True
    if s in ("false", "no", "0", "backward", "event_b_before_event_a", "b_before_a"):
        return False
    return None


def normalize_confidence(value):
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("very high", "high", "strong"):
            return 0.85
        if lower in ("medium", "moderate"):
            return 0.65
        if lower in ("low", "weak"):
            return 0.4

    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.5

    # Fix model outputs like 95.0 or 95.7.
    if 1.0 < x <= 100.0:
        x = x / 100.0

    return max(0.0, min(1.0, x))


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    # Keep the latest output for each scenario/step.
    latest = {}

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            try:
                key = (record.get("scenario_id", ""), int(record.get("step", 0)))
            except Exception:
                continue

            latest[key] = {**record, "_line_no": line_no}

    rows = []

    for key in sorted(latest.keys()):
        record = latest[key]
        response = record.get("response", "")
        parsed = extract_json_object(response)

        parse_ok = parsed is not None
        parse_error = "" if parse_ok else "could_not_parse_json"

        supports_forward = None
        confidence = 0.5

        if parse_ok:
            supports_forward = normalize_bool(parsed.get("supports_forward"))
            confidence = normalize_confidence(parsed.get("confidence", 0.5))

            if supports_forward is None:
                parse_ok = False
                parse_error = "missing_or_invalid_supports_forward"

        rows.append({
            "scenario_id": record.get("scenario_id", ""),
            "step": record.get("step", ""),
            "model": record.get("model", ""),
            "evidence": record.get("evidence", ""),
            "supports_forward": "" if supports_forward is None else str(supports_forward),
            "confidence": confidence,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "raw_response": "" if parse_ok else response,
            "raw_line_no": record.get("_line_no", ""),
        })

    fieldnames = [
        "scenario_id",
        "step",
        "model",
        "evidence",
        "supports_forward",
        "confidence",
        "parse_ok",
        "parse_error",
        "raw_response",
        "raw_line_no",
    ]

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if str(r["parse_ok"]).lower() == "true")
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Parse OK: {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
