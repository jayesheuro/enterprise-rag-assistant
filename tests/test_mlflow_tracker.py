import pytest
import os
import tempfile
from unittest.mock import patch
from app.core.config import AppConfig, MLflowConfig
from app.services.tracking.mlflow_tracker import get_tracker, MLflowTracker, NoOpTracker
import mlflow

def test_noop_behavior():
    """Test that NoOpTracker is used when disabled, and does nothing."""
    config = AppConfig(mlflow=MLflowConfig(enabled=False))
    tracker = get_tracker(config)
    assert isinstance(tracker, NoOpTracker)
    
    # Should not raise any errors
    tracker.log_eval_run(None)
    tracker.sync_prompt("test", "1.0", "test template", "msg")
    assert tracker._flatten_config({"a": 1}) == {}
    assert tracker._get_git_commit() == ""

def test_param_flattening():
    """Test that _flatten_config works correctly."""
    config = AppConfig(mlflow=MLflowConfig(enabled=True))
    with patch('app.services.tracking.mlflow_tracker.HAS_MLFLOW', True):
        tracker = get_tracker(config)
        assert isinstance(tracker, MLflowTracker)
        
        test_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": "3"
                }
            }
        }
        
        flat = tracker._flatten_config(test_dict)
        assert flat == {
            "a": "1",
            "b.c": "2",
            "b.d.e": "3"
        }

def test_prompt_sync_idempotency(tmp_path):
    """Test that prompt sync creates only one version if template is unchanged."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    config = AppConfig(mlflow=MLflowConfig(enabled=True, tracking_uri=tracking_uri))
    
    with patch('app.services.tracking.mlflow_tracker.HAS_MLFLOW', True):
        tracker = get_tracker(config)
        
        # Sync first time
        tracker.sync_prompt("test_prompt", "1.0.0", "Hello {name}", "init")
        
        # Verify it was registered
        prompts = mlflow.genai.search_prompts("name='test_prompt'")
        assert len(prompts) == 1
        
        # Sync second time with same template
        tracker.sync_prompt("test_prompt", "1.0.0", "Hello {name}", "init")
        
        # Should still be 1 (No exception raised)
        
        # Sync with different template
        tracker.sync_prompt("test_prompt", "2.0.0", "Hi {name}", "update")

