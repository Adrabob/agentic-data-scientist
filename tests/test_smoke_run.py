import os
import subprocess
import sys


def test_smoke_run_creates_outputs(tmp_path):
    out_dir = tmp_path / "agt_outputs"
    cmd = [
        sys.executable, "run_agent.py",
        "--data", "data/demo.csv",
        "--target", "auto",
        "--output_root", str(out_dir),
    ]
    # Run the script; it should complete successfully
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"run_agent failed: {res.stderr}"
    assert os.path.exists(str(out_dir)), "Output directory was not created"
