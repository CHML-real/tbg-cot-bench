import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IN_PATH = RESULTS / "ollama_cumulative_v4_raw.jsonl"
OUT_PATH = RESULTS / "ollama_cumulative_v4.csv"


def clean_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = candidate.replace("True", "true").replace("False", "false")
    candidate = candidate.replace("None", "null")
    candidate = re.sub(r'([{,]\s*)(order|confidence)\s*:', r'\1"\2":', candidate)
    return candidate


def find_json_candidates(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for item in fenced:
        yield item

    objects = re.findall(r"\{[^{}]*(?:order|confidence)[^{}]*\}", text, flags=re.IGNORECASE | re.DOTALL)
    for item in objects:
        yield item

    if text.startswith("{") and text.endswith("}"):
        yield text


def extract_json_object(text: str):
    for candidate in find_json_candidates(text):
        fixed = clean_candidate(candidate)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_order(value):
    if value is None:
        return None

    s = str(value).strip().upper()
    s = s.replace("-", "_").replace(" ", "_")

    if s in ("A_BEFORE_B", "EVENT_A_BEFORE_EVENT_B", "FORWARD", "TRUE"):
        return "A_BEFORE_B"
    if s in ("B_BEFORE_A", "EVENT_B_BEFORE_EVENT_A", "BACKWARD", "FALSE"):
        return "B_BEFORE_A"
    if s in ("UNCLEAR", "AMBIGUOUS", "UNKNOWN", "UNDETERMINED", "NEUTRAL"):
        return "UNCLEAR"

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

    if 1.0 < x <= 100.0:
        x = x / 100.0

    return max(0.0, min(1.0, x))


def order_to_p(order: str, confidence: float):
    if order == "A_BEFORE_B":
        return confidence
    if order == "B_BEFORE_A":
        return 1.0 - confidence
    return 0.5


def order_to_verdict(order: str):
    if order == "A_BEFORE_B":
        return "forward"
    if order == "B_BEFORE_A":
        return "backward"
    return "ambiguous"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

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

        order = None
        confidence = 0.5

        if parse_ok:
            order = normalize_order(parsed.get("order"))
            confidence = normalize_confidence(parsed.get("confidence", 0.5))

            if order is None:
                parse_ok = False
                parse_error = "missing_or_invalid_order"

        p_forward = order_to_p(order, confidence) if parse_ok else 0.5
        verdict = order_to_verdict(order) if parse_ok else "ambiguous"

        rows.append({
            "scenario_id": record.get("scenario_id", ""),
            "step": record.get("step", ""),
            "model": record.get("model", ""),
            "num_evidence_items": record.get("num_evidence_items", ""),
            "order": "" if order is None else order,
            "confidence": confidence,
            "p_forward": round(p_forward, 6),
            "verdict": verdict,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "raw_response": "" if parse_ok else response,
            "raw_line_no": record.get("_line_no", ""),
        })

    fieldnames = [
        "scenario_id",
        "step",
        "model",
        "num_evidence_items",
        "order",
        "confidence",
        "p_forward",
        "verdict",
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
