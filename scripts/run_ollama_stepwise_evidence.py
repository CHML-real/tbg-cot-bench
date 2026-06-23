import json
import os
import urllib.request
from pathlib import Path

MODEL = os.environ.get("OLLAMA_MODEL", "exaone-local")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"
OUT = RESULTS / "ollama_stepwise_evidence_raw.jsonl"
RESULTS.mkdir(parents=True, exist_ok=True)


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=300) as res:
        obj = json.loads(res.read().decode("utf-8"))
        return obj.get("response", "")


def make_prompt(event_a: str, event_b: str, evidence: str) -> str:
    return f"""
Return ONLY valid JSON.

Task:
Judge whether the evidence supports this temporal claim:

"Event A occurred before Event B."

Event A:
{event_a}

Event B:
{event_b}

Evidence:
{evidence}

Use exactly this JSON schema:
{{"supports_forward": true, "confidence": 0.0}}

Rules:
- supports_forward must be true if the evidence supports Event A before Event B.
- supports_forward must be false if the evidence supports Event B before Event A.
- confidence must be a number from 0.0 to 1.0.
- Do not write markdown.
- Do not explain.
- Do not add extra keys.
""".strip()


def main():
    scenario_files = sorted(SCENARIOS.glob("sc*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {SCENARIOS}")

    with OUT.open("w", encoding="utf-8") as f:
        for path in scenario_files:
            scenario = json.loads(path.read_text(encoding="utf-8"))
            sid = scenario["id"]
            event_a = scenario["events"]["event_a"]["label"]
            event_b = scenario["events"]["event_b"]["label"]

            for idx, step in enumerate(scenario["steps"], start=1):
                evidence = step["text"]
                print(f"Running {sid} step {idx} with {MODEL}...")

                prompt = make_prompt(event_a, event_b, evidence)
                response = call_ollama(prompt)

                row = {
                    "scenario_id": sid,
                    "step": idx,
                    "model": MODEL,
                    "event_a": event_a,
                    "event_b": event_b,
                    "evidence": evidence,
                    "response": response,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
