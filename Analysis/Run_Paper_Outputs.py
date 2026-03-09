from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


SCRIPT_ORDER = [
    "DP_vs_TFG_V_Sweep.py",
    "Policy_GridSearch_vs_TFG.py",
    "Premium_Priority_Table.py",
    "Journal_Experiments.py",
]


def main() -> int:
    analysis_dir = Path(__file__).resolve().parent
    repo_root = analysis_dir.parent
    py_exe = sys.executable
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    for script_name in SCRIPT_ORDER:
        script_path = analysis_dir / script_name
        if not script_path.exists():
            print(f"[pipeline] missing script: {script_path}", flush=True)
            return 1
        print(f"[pipeline] running {script_name}", flush=True)
        result = subprocess.run([py_exe, str(script_path)], check=False, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            print(f"[pipeline] failed: {script_name} (code={result.returncode})", flush=True)
            return int(result.returncode)
    print("[pipeline] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
