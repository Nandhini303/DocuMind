"""
DocuMind Project Entrypoint
----------------------------
Can be run from root: python main.py
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def main():
    import uvicorn
    from app.core.config import settings
    print(f"Starting {settings.APP_NAME} on http://localhost:{settings.PORT} ...")
    uvicorn.run("main:app", app_dir=str(backend_dir), host="0.0.0.0", port=settings.PORT, reload=True)


if __name__ == "__main__":
    main()
