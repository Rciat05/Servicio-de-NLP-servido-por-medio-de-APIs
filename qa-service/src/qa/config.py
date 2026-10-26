"""Settings. Everything sensitive comes from environment variables, never from code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Folder with the fine-tuned model (default: <repo>/models/final)."""
    p = Path(os.environ.get("MODEL_DIR", "models/final"))
    return p if p.is_absolute() else ROOT / p


def team_name() -> str:
    return os.environ.get("QA_TEAM", "team-XX")


def api_token() -> str:
    """Read on every call so a missing token means 'reject everything'."""
    return os.environ.get("QA_API_TOKEN", "")
