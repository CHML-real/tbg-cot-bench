import json
import os
import urllib.request
from pathlib import Path

MODEL = os.environ.get("OLLAMA_MODEL", "exaone-local")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
OUT = ROOT / "results" / "ollama_exaone_cot.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096
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


def make_prompt(scenario: dict) -> str:
    event_a = scenario["events"]["event_a"]["label"]
    event_b = scenario["events"]["event_b"]["label"]

    evidence_lines = "\n".join(
        f"{i + 1}. {step['text']}"
        for i, step in enumerate(scenario["steps"])
    )

    return f"""
You are evaluating temporal event ordering.

Task:
Determine whether Event A occurred before Event B.

Event A:
{event_a}

Event B:
{event_b}

Evidence:
{evidence_lines}

Instructions:
- Evaluate each evidence item one by one.
- For each step, decide whether it supports Event A before Event B.
- Return only valid JSON.
- Do not include markdown.
- Do not include commentary outside JSON.
- confidence must be a number between 0.0 and 1.0.

JSON schema:
{{
  "steps": [
    {{
      "text": "short reasoning step",
      "supports_forward": true,
      "confidence": 0.0
    }}
  ],
  "final_answer": "forward"
}}

Allowed final_answer values:
forward, backward, ambiguous
""".strip()


def main():
    files = sorted(SCENARIOS.glob("sc*.json"))

    if not files:
        raise FileNotFoundError(f"No scenario files found in {SCENARIOS}")

    with OUT.open("w", encoding="utf-8") as f:
        for path in files:
            scenario = json.loads(path.read_text(encoding="utf-8"))
            sid = scenario["id"]

            print(f"Running {sid} with {MODEL}...")

            prompt = make_prompt(scenario)
            response = call_ollama(prompt)

            row = {
                "scenario_id": sid,
                "model": MODEL,
                "response": response
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
