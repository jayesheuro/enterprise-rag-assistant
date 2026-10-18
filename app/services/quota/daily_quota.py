"""Global daily quota tracker for the public demo deployment.

Implements a simple file-based counter that limits total chat queries
across ALL sessions per calendar day (UTC). Designed to protect
Gemini free-tier quotas on Hugging Face Spaces.

Design: file-based to survive process restarts within a single container
lifetime. HF free-tier disk is ephemeral (reset on restart), which
coincidentally resets the quota counter on restart too — acceptable.

Environment variables:
    DEMO_MODE        - "true" to enable quota enforcement
    DEMO_DAILY_CAP   - max queries per UTC day (default: 50)
    QUOTA_FILE_PATH  - path to the counter file (default: /tmp/daily_quota.json)
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
DEMO_DAILY_CAP: int = int(os.getenv("DEMO_DAILY_CAP", "50"))
QUOTA_FILE_PATH: Path = Path(os.getenv("QUOTA_FILE_PATH", "/tmp/daily_quota.json"))

# Thread lock to make counter updates safe under uvicorn's multi-threaded workers
_lock = threading.Lock()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _today_utc() -> str:
    """Return today's date in UTC as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_state() -> dict:
    """Read the quota state file. Returns empty state if missing or corrupt."""
    try:
        if QUOTA_FILE_PATH.exists():
            with open(QUOTA_FILE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read quota file (%s); starting fresh.", e)
    return {"date": _today_utc(), "count": 0}


def _write_state(state: dict) -> None:
    """Persist the quota state file atomically."""
    try:
        QUOTA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = QUOTA_FILE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        tmp.replace(QUOTA_FILE_PATH)
    except OSError as e:
        logger.error("Could not write quota file: %s", e)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_quota_status() -> dict:
    """Return current quota state without consuming a slot.

    Returns:
        dict with keys: date, count, cap, remaining, exhausted
    """
    if not DEMO_MODE:
        return {"date": _today_utc(), "count": 0, "cap": None, "remaining": None, "exhausted": False}

    with _lock:
        state = _read_state()
        today = _today_utc()
        # Roll over on new day
        if state.get("date") != today:
            state = {"date": today, "count": 0}

    remaining = max(0, DEMO_DAILY_CAP - state["count"])
    return {
        "date": state["date"],
        "count": state["count"],
        "cap": DEMO_DAILY_CAP,
        "remaining": remaining,
        "exhausted": remaining == 0,
    }


def consume_quota() -> dict:
    """Try to consume one quota slot.

    Returns:
        dict with keys: allowed (bool), count, cap, remaining, exhausted, message
    """
    if not DEMO_MODE:
        return {"allowed": True, "count": 0, "cap": None, "remaining": None, "exhausted": False, "message": ""}

    with _lock:
        state = _read_state()
        today = _today_utc()

        # Roll over on new day
        if state.get("date") != today:
            state = {"date": today, "count": 0}

        if state["count"] >= DEMO_DAILY_CAP:
            # Quota exhausted — do not increment
            _write_state(state)
            remaining = 0
            return {
                "allowed": False,
                "count": state["count"],
                "cap": DEMO_DAILY_CAP,
                "remaining": remaining,
                "exhausted": True,
                "message": (
                    f"🚦 **Demo quota reached for today** ({DEMO_DAILY_CAP} questions/day). "
                    "This keeps the public demo free and available for everyone. "
                    "Want unlimited use? **Clone the repo and run it locally with Ollama** — "
                    "zero API costs, no caps. See the README quickstart. "
                    "Quota resets at UTC midnight."
                ),
            }

        # Allow and increment
        state["count"] += 1
        _write_state(state)
        remaining = max(0, DEMO_DAILY_CAP - state["count"])

    logger.info("Demo quota consumed: %d/%d today.", state["count"], DEMO_DAILY_CAP)
    return {
        "allowed": True,
        "count": state["count"],
        "cap": DEMO_DAILY_CAP,
        "remaining": remaining,
        "exhausted": False,
        "message": "",
    }

