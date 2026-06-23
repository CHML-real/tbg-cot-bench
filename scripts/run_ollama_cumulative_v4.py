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
OUT = RESULTS / "ollama_cumulative_v4_raw.jsonl"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_completed():
    completed = set()
    if not OUT.exists():
        return completed

    with OUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add((row.get("scenario_id"), int(row.get("step"))))
            except Exception:
                continue
    return completed


def call_ollama(prompt: str, retries: int = 2, sleep_sec: float = 1.0) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 4096
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


def make_prompt(event_a: str, event_b: str, evidence_items: list[str]) -> str:
    evidence_text = "\n".join(
        f"{i + 1}. {item}" for i, item in enumerate(evidence_items)
    )

    return f"""
Return one JSON object only.

You are tracking belief about temporal event order as evidence accumulates.

Claim:
Event A occurred before Event B.

Event A: {event_a}
Event B: {event_b}

Evidence available so far:
{evidence_text}

Choose the current best temporal conclusion after considering all evidence above.

Allowed order values:
- "A_BEFORE_B": current evidence supports Event A before Event B.
- "B_BEFORE_A": current evidence supports Event B before Event A.
- "UNCLEAR": current evidence is insufficient or mixed.

Output exactly this JSON schema:
{{"order": "A_BEFORE_B", "confidence": 0.0}}

Rules:
- Compare the two named events, not isolated words like "before" or "after".
- Use all evidence available so far.
- confidence must be from 0.0 to 1.0, not 0 to 100.
- No markdown.
- No explanation.
- No extra keys.
""".strip()


def iter_cumulative_steps():
    files = sorted(SCENARIOS.glob("sc*.json"))
    if not files:
        raise FileNotFoundError(f"No scenario files found in {SCENARIOS}")

    for path in files:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        sid = scenario["id"]
        event_a = scenario["events"]["event_a"]["label"]
        event_b = scenario["events"]["event_b"]["label"]
        steps = [s["text"] for s in scenario["steps"]]

        for idx in range(1, len(steps) + 1):
            yield {
                "scenario_id": sid,
                "step": idx,
                "event_a": event_a,
                "event_b": event_b,
                "evidence_items": steps[:idx],
            }


def main():
    completed = load_completed()
    tasks = [
        item for item in iter_cumulative_steps()
        if (item["scenario_id"], item["step"]) not in completed
    ]

    print(f"Model: {MODEL}")
    print(f"Tasks to run: {len(tasks)}")

    if not tasks:
        print("Nothing to run.")
        return

    with OUT.open("a", encoding="utf-8") as f:
        for item in tasks:
            sid = item["scenario_id"]
            step = item["step"]
            print(f"Running {sid} cumulative step {step} with {MODEL}...")

            prompt = make_prompt(
                item["event_a"],
                item["event_b"],
                item["evidence_items"],
            )
            response = call_ollama(prompt)

            row = {
                "scenario_id": sid,
                "step": step,
                "model": MODEL,
                "event_a": item["event_a"],
                "event_b": item["event_b"],
                "num_evidence_items": len(item["evidence_items"]),
                "evidence_items": item["evidence_items"],
                "response": response,
                "runner": "cumulative_v4_json_mode",
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved/appended: {OUT}")


if __name__ == "__main__":
    main()
