import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "tbg_cot_experiment_report.md"


def test_generate_experiment_report_script_runs():
    script = ROOT / "scripts" / "generate_experiment_report.py"
    assert script.exists(), "generate_experiment_report.py is missing"

    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=True,
    )

    assert REPORT.exists(), "report was not generated"

    text = REPORT.read_text(encoding="utf-8")
    assert "TBG-CoT-Bench Local Experiment Report" in text
    assert "EXAONE step-wise evidence v2.1" in text
    assert "application-level benchmark" in text
