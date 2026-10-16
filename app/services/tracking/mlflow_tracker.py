"""MLflow integration with graceful degradation."""

import subprocess
from typing import Any

from app.core.config import AppConfig
from app.core.logging import get_logger
from app.services.evaluation.report import EvalReport

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.genai
    HAS_MLFLOW = True
except ImportError:
    mlflow: Any = None  # type: ignore
    HAS_MLFLOW = False


class NoOpTracker:
    """A tracker that does nothing, used when MLflow is disabled or unavailable."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        logger.info("MLflow tracking is disabled or not installed.")

    def log_eval_run(self, report: EvalReport, tags: dict[str, str] | None = None) -> None:
        """No-op."""
        pass

    def sync_prompt(
        self, name: str, version: str, template: str, commit_message: str
    ) -> None:
        """No-op."""
        pass

    def _flatten_config(self, config_snapshot: dict[str, Any]) -> dict[str, str]:
        """No-op."""
        return {}

    def _get_git_commit(self) -> str:
        """No-op."""
        return ""


class MLflowTracker:
    """Tracker implementation that sends data to MLflow."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        logger.info(f"Initialized MLflow tracker at {self.config.mlflow.tracking_uri}")

    def _flatten_config(self, config_snapshot: dict[str, Any], prefix: str = "") -> dict[str, str]:
        """Flatten a nested dictionary to a single level with dot notation."""
        items: list[tuple[str, Any]] = []
        for k, v in config_snapshot.items():
            new_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, f"{new_key}.").items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def _get_git_commit(self) -> str:
        """Get the current Git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def log_eval_run(self, report: EvalReport, tags: dict[str, str] | None = None) -> None:
        """Log an evaluation run to MLflow."""
        with mlflow.start_run(run_name=report.run_id):
            # Set tags
            run_tags = {
                "run_id": report.run_id,
                "dataset_version": report.dataset_version,
                "retrieval_only": str(report.config_snapshot.get("retrieval_only", False)),
                "git_commit": self._get_git_commit(),
            }
            if tags:
                run_tags.update(tags)
            mlflow.set_tags(run_tags)

            # Log parameters
            params = self._flatten_config(report.config_snapshot)
            mlflow.log_params(params)

            # Log metrics
            metrics = {
                **report.aggregate_retrieval_metrics,
                **report.aggregate_ragas_metrics,
            }
            if not report.retrieval_only:
                metrics["refusal_accuracy"] = report.aggregate_refusal_accuracy
            # MLflow metric names cannot contain '@'
            sanitized_metrics = {
                k.replace("@", "_at_"): v for k, v in metrics.items() if v is not None
            }
            mlflow.log_metrics(sanitized_metrics)

            # Log artifacts (both JSON and Markdown)
            mlflow.log_dict(report.model_dump(mode="json"), "report.json")
            try:
                mlflow.log_text(report.to_markdown(), "report.md")
            except Exception as e:
                logger.warning(f"Could not log report.md artifact: {e}")

            logger.info(f"Logged eval run {report.run_id} to MLflow.")

    def sync_prompt(
        self, name: str, version: str, template: str, commit_message: str
    ) -> None:
        """Register a prompt if it doesn't already exist with the same template."""
        # Find existing prompts
        try:
            # Check if this prompt and version exists
            _ = mlflow.genai.search_prompts(f"name='{name}'")
            # This doesn't filter by version, so we check the latest or all
            # Just iterating through available prompts is not supported directly in the search.
            # For simplicity, let's check if we can get the prompt.
            # Using mlflow.genai.get_prompt will raise if it doesn't exist.
            try:
                existing = mlflow.genai.load_prompt(name_or_uri=name)
                if existing.template == template:
                    logger.info(f"Prompt {name} version {version} already matches. Skipping.")
                    return
            except Exception:
                pass  # version doesn't exist, we can create
        except Exception:
            pass # prompt doesn't exist

        # Register prompt
        mlflow.genai.register_prompt(
            name=name,
            template=template,
            tags={"version": version, "commit_message": commit_message},
        )
        logger.info(f"Registered prompt {name} version {version} to MLflow.")


def get_tracker(config: AppConfig) -> MLflowTracker | NoOpTracker:
    """Factory function to get the appropriate tracker instance."""
    if HAS_MLFLOW and config.mlflow.enabled:
        return MLflowTracker(config)
    return NoOpTracker(config)
