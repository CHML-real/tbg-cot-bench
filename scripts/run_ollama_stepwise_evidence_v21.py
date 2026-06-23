import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("OLLAMA_MODEL", "exaone-local")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

RAW_OUT = RESULTS / "ollama_stepwise_evidence_raw.jsonl"
PARSED_CURRENT = RESULTS / "ollama_stepwise_evidence.csv"

RESULTS.mkdir(parents=True, exist_ok=True)


def load_completed_from_raw():
    completed = set()
    if not RAW_OUT.exists():
        return completed

    with RAW_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add((row.get("scenario_id"), int(row.get("step"))))
            except Exception:
                continue

    return completed


def load_failed_from_parsed():
    failed = set()
    if not PARSED_CURRENT.exists():
        return failed

    import csv
    with PARSED_CURRENT.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = row["scenario_id"]
                step = int(row["step"])
            except Exception:
                continue

            parse_ok = str(row.get("parse_ok", "")).lower() == "true"
            if not parse_ok:
                failed.add((sid, step))

    return failed


def call_ollama(prompt: str, retries: int = 2, sleep_sec: float = 1.0) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 2048
        }
    }

    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=300) as res:
                obj = json.loads(res.read().decode("utf-8"))
                return obj.get("response", "")
        except Exception as e:
            last_error = e
            print(f"  retry {attempt + 1}/{retries + 1} failed: {e}")
            time.sleep(sleep_sec)

    raise RuntimeError(f"Ollama call failed after retries: {last_error}")


def make_prompt(event_a: str, event_b: str, evidence: str) -> str:
    return f"""
Return one JSON object only.

Decide whether the evidence supports the claim:
Event A occurred before Event B.

Event A: {event_a}
Event B: {event_b}
Evidence: {evidence}

Output schema:
{{"supports_forward": true, "confidence": 0.0}}

Rules:
- supports_forward=true means the evidence supports Event A before Event B.
- supports_forward=false means the evidence supports Event B before Event A.
- confidence must be between 0.0 and 1.0, not 0 to 100.
- No markdown.
- No explanation.
- No extra keys.
""".strip()


def iter_scenario_steps():
    scenario_files = sorted(SCENARIOS.glob("sc*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {SCENARIOS}")

    for path in scenario_files:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        sid = scenario["id"]
        event_a = scenario["events"]["event_a"]["label"]
        event_b = scenario["events"]["event_b"]["label"]

        for idx, step in enumerate(scenario["steps"], start=1):
            yield {
                "scenario_id": sid,
                "step": idx,
                "event_a": event_a,
                "event_b": event_b,
                "evidence": step["text"],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["missing", "failed", "all"],
        default="missing",
        help="missing: only steps not in raw jsonl; failed: missing + parsed failures; all: rerun everything into a new raw file"
    )
    args = parser.parse_args()

    if args.mode == "all":
        if RAW_OUT.exists():
            backup = RAW_OUT.with_suffix(".jsonl.bak")
            RAW_OUT.rename(backup)
            print(f"Backed up existing raw file to: {backup}")
        completed = set()
        failed = set()
    else:
        completed = load_completed_from_raw()
        failed = load_failed_from_parsed() if args.mode == "failed" else set()

    tasks = []
    for item in iter_scenario_steps():
        key = (item["scenario_id"], item["step"])

        if args.mode == "all":
            tasks.append(item)
        elif args.mode == "missing":
            if key not in completed:
                tasks.append(item)
        elif args.mode == "failed":
            if key not in completed or key in failed:
                tasks.append(item)

    print(f"Model: {MODEL}")
    print(f"Mode: {args.mode}")
    print(f"Tasks to run: {len(tasks)}")

    if not tasks:
        print("Nothing to run.")
        return

    with RAW_OUT.open("a", encoding="utf-8") as f:
        for item in tasks:
            sid = item["scenario_id"]
            step = item["step"]
            print(f"Running {sid} step {step} with {MODEL}...")

            prompt = make_prompt(item["event_a"], item["event_b"], item["evidence"])
            response = call_ollama(prompt)

            row = {
                "scenario_id": sid,
                "step": step,
                "model": MODEL,
                "event_a": item["event_a"],
                "event_b": item["event_b"],
                "evidence": item["evidence"],
                "response": response,
                "runner": "v2.1_json_mode",
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved/appended: {RAW_OUT}")


if __name__ == "__main__":
    main()
