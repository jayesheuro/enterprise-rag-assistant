"""Evaluation API routes.

Provides endpoints to list and retrieve evaluation reports.
"""

import json

from fastapi import APIRouter, HTTPException

from app.core.config import PROJECT_ROOT

router = APIRouter(prefix="/eval", tags=["evaluation"])

REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


@router.get("/reports")
async def list_reports() -> dict[str, list[str]]:
    """List all available evaluation report IDs."""
    if not REPORTS_DIR.exists():
        return {"reports": []}

    reports = []
    for file_path in REPORTS_DIR.glob("*.json"):
        if not file_path.name.endswith("_checkpoint.json"):
            reports.append(file_path.stem)

    return {"reports": sorted(reports, reverse=True)}


@router.get("/reports/{run_id}")
async def get_report(run_id: str) -> dict:
    """Get the full JSON report for a specific run ID."""
    report_path = REPORTS_DIR / f"{run_id}.json"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Report file is corrupted")


@router.get("/narrative")
async def get_experiment_narrative() -> dict[str, str]:
    """Get the markdown content of the experiment narrative (experiment_1.md)."""
    narrative_path = REPORTS_DIR / "experiment_1.md"
    if not narrative_path.exists():
        return {"content": "No experiment narrative found."}

    try:
        with open(narrative_path, encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read narrative: {e}")
