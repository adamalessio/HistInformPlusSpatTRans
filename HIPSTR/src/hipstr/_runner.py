from __future__ import annotations

import runpy
import sys
from pathlib import Path

def run_bundled_script(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Bundled script not found: {script_path}")
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")
