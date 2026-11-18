"""
DocuMind AI — Streamlit App Launcher (Backend Directory)
-------------------------------------------------------
Enables running: `streamlit run streamlit_app.py` directly from inside the `backend` folder.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
root_dir = backend_dir.parent

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute root streamlit_app.py
app_path = root_dir / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code, globals())
