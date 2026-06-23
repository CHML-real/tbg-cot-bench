import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IN_PATH = RESULTS / "ollama_stepwise_evidence_raw.jsonl"
OUT_PATH = RESULTS / "ollama_stepwise_evidence.csv"


def extract_json_object(text: str):
    text = (text or "").strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        candidate = candidate.replace("True", "true").replace("False", "false")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

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
        if lower in ("high", "strong"):
            return 0.85
        if lower in ("medium", "moderate"):
            return 0.65
        if lower in ("low", "weak"):
            return 0.4

    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.5

    return max(0.0, min(1.0, x))


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    rows = []

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
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
