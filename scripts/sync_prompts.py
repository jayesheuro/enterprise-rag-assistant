"""Sync prompt templates from YAML files to MLflow prompt registry.

Reads every prompts/*.yaml, registers or updates each prompt in MLflow.
Idempotent: unchanged templates do not create new versions.

Usage:
    python scripts/sync_prompts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.generation.prompt_builder import load_prompt
from app.services.tracking.mlflow_tracker import get_tracker


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)
    config = get_settings()
    tracker = get_tracker(config)

    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    yaml_files = sorted(prompts_dir.glob("*.yaml"))

    if not yaml_files:
        logger.warning("No prompt YAML files found in prompts/")
        return

    logger.info(f"Found {len(yaml_files)} prompt files")

    for yaml_file in yaml_files:
        prompt_name = yaml_file.stem
        try:
            prompt = load_prompt(prompt_name)
            # Build commit message from latest changelog entry
            commit_msg = ""
            if prompt.changelog:
                latest = (
                    prompt.changelog[-1]
                    if isinstance(prompt.changelog[-1], dict)
                    else prompt.changelog[0]
                )
                commit_msg = latest.get("changes", f"Version {prompt.version}")

            tracker.sync_prompt(
                name=prompt.name,
                version=prompt.version,
                template=prompt.template,
                commit_message=commit_msg,
            )
        except Exception as e:
            logger.error(f"Failed to sync prompt '{prompt_name}': {e}")


if __name__ == "__main__":
    main()
