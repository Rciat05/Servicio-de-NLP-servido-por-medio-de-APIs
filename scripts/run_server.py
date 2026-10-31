"""Start the service on 127.0.0.1:8000 (the tunnel talks to it locally).

Usage (from the repo root, venv active):   python scripts/run_server.py
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "qa.server:app",
        host=os.environ.get("QA_HOST", "127.0.0.1"),
        port=8000,
        workers=1,
        log_level="info",
    )
