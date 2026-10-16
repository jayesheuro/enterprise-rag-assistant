"""Backfill MLflow with existing evaluation reports.

Reads JSON reports from eval/reports/ and logs them as MLflow runs,
enabling comparison of historical experiment results in the MLflow UI.

Usage:
    python scripts/backfill_mlflow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.evaluation.report import REPORTS_DIR, EvalReport
from app.services.tracking.mlflow_tracker import get_tracker


def main() -> None:
    """Run the MLflow backfill process."""
    setup_logging()
    logger = get_logger(__name__)
    config = get_settings()
    tracker = get_tracker(config)

    report_files = sorted(REPORTS_DIR.glob("run_*.json"))

    if not report_files:
        logger.warning("No run_*.json reports found in eval/reports/")
        return

    logger.info(f"Found {len(report_files)} reports to backfill")

    for report_file in report_files:
        run_id = report_file.stem
        try:
            report = EvalReport.load(run_id)
            tracker.log_eval_run(report, tags={"experiment_group": "phase4_experiment"})
            logger.info(f"Backfilled {run_id}")
        except Exception as e:
            logger.error(f"Failed to backfill {run_id}: {e}")


if __name__ == "__main__":
    main()
