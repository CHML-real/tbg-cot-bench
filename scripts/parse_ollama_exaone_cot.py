import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IN_PATH = RESULTS / "ollama_exaone_cot.jsonl"
OUT_PATH = RESULTS / "ollama_evidence.csv"


def extract_json_object(text: str):
    text = text.strip()
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
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    s = str(value).strip().lower()
    if s in ("true", "yes", "1", "forward", "event_a_before_event_b"):
        return True
    if s in ("false", "no", "0", "backward", "event_b_before_event_a"):
        return False
    return None


def normalize_confidence(value):
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
            scenario_id = record.get("scenario_id", "")
            model = record.get("model", "")
            response = record.get("response", "")

            parsed = extract_json_object(response)
            if parsed is None:
                rows.append({
                    "scenario_id": scenario_id,
                    "step": "",
                    "model": model,
                    "text": "",
                    "supports_forward": "",
                    "confidence": "",
                    "final_answer": "",
                    "parse_ok": False,
                    "parse_error": "could_not_parse_json",
                    "raw_response": response,
                })
                continue

            steps = parsed.get("steps", [])
            final_answer = parsed.get("final_answer", "")

            if not isinstance(steps, list):
                rows.append({
                    "scenario_id": scenario_id,
                    "step": "",
                    "model": model,
                    "text": "",
                    "supports_forward": "",
                    "confidence": "",
                    "final_answer": final_answer,
                    "parse_ok": False,
                    "parse_error": "steps_not_list",
                    "raw_response": response,
                })
                continue

            for i, step_obj in enumerate(steps, start=1):
                if not isinstance(step_obj, dict):
                    step_obj = {"text": str(step_obj)}

                supports_forward = normalize_bool(step_obj.get("supports_forward"))
                confidence = normalize_confidence(step_obj.get("confidence", 0.5))

                rows.append({
                    "scenario_id": scenario_id,
                    "step": i,
                    "model": model,
                    "text": step_obj.get("text", ""),
                    "supports_forward": "" if supports_forward is None else str(supports_forward),
                    "confidence": confidence,
                    "final_answer": final_answer,
                    "parse_ok": True,
                    "parse_error": "",
                    "raw_response": "",
                })

    RESULTS.mkdir(exist_ok=True)

    fieldnames = [
        "scenario_id",
        "step",
        "model",
        "text",
        "supports_forward",
        "confidence",
        "final_answer",
        "parse_ok",
        "parse_error",
        "raw_response",
    ]

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
