"""
Optional sanity test.
Run:
    python tests/sanity_check.py
from repo root (after installing requirements).
"""

import os
import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data", "data/example_dataset.csv",
        "--target", "auto",
        "--quiet",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise SystemExit("Sanity check failed.")

    out_dir = result.stdout.strip().splitlines()[-1].strip()
    print("Output dir:", out_dir)

    expected = ["report.md", "metrics.json", "reflection.json", "eda_summary.json", "confusion_matrix.png"]
    missing = [f for f in expected if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        raise SystemExit(f"Missing expected outputs: {missing}")

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
