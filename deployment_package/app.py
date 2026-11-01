"""Root entrypoint for EdgeInspect AI deployment on Streamlit Cloud & Hugging Face Spaces."""
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
APP_DIR = ROOT_DIR / "app"

for p in [ROOT_DIR, SRC_DIR, APP_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Execute the main dashboard script whether located in app/ or directly at root
dashboard_path = APP_DIR / "dashboard.py"
if not dashboard_path.exists():
    dashboard_path = ROOT_DIR / "dashboard.py"

if not dashboard_path.exists():
    raise FileNotFoundError(f"Could not find dashboard.py in {APP_DIR} or {ROOT_DIR}")

runpy.run_path(str(dashboard_path), run_name="__main__")
