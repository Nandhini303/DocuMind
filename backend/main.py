"""
DocuMind AI — Server Entrypoint
--------------------------------
Exposes the FastAPI application instance for Uvicorn and ASGI servers.
Supports:
    uvicorn main:app --reload --port 8080
    python main.py
"""

import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
from app.core.config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
