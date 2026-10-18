"""Unit tests for the daily demo quota tracker."""

import json
from unittest.mock import patch

from app.services.quota import daily_quota


def test_quota_disabled_when_demo_mode_false(tmp_path):
    """When DEMO_MODE is False, quota should never restrict requests."""
    fake_quota_file = tmp_path / "quota.json"
    with patch.object(daily_quota, "DEMO_MODE", False), \
         patch.object(daily_quota, "QUOTA_FILE_PATH", fake_quota_file):
        status = daily_quota.get_quota_status()
        assert status["exhausted"] is False
        assert status["cap"] is None

        # Consuming should always be allowed
        res = daily_quota.consume_quota()
        assert res["allowed"] is True
        assert res["exhausted"] is False
        assert not fake_quota_file.exists()


def test_quota_enforcement_and_exhaustion(tmp_path):
    """When DEMO_MODE is True, enforce cap and return friendly message once exhausted."""
    fake_quota_file = tmp_path / "quota.json"
    test_cap = 3

    with patch.object(daily_quota, "DEMO_MODE", True), \
         patch.object(daily_quota, "DEMO_DAILY_CAP", test_cap), \
         patch.object(daily_quota, "QUOTA_FILE_PATH", fake_quota_file):

        # Initial status
        status = daily_quota.get_quota_status()
        assert status["count"] == 0
        assert status["remaining"] == 3
        assert status["exhausted"] is False

        # Consume 1
        res1 = daily_quota.consume_quota()
        assert res1["allowed"] is True
        assert res1["count"] == 1
        assert res1["remaining"] == 2

        # Consume 2
        res2 = daily_quota.consume_quota()
        assert res2["allowed"] is True
        assert res2["count"] == 2

        # Consume 3 (hits cap)
        res3 = daily_quota.consume_quota()
        assert res3["allowed"] is True
        assert res3["count"] == 3
        assert res3["remaining"] == 0
        assert res3["exhausted"] is False

        # Attempt 4 (exceeds cap)
        res4 = daily_quota.consume_quota()
        assert res4["allowed"] is False
        assert res4["count"] == 3
        assert res4["remaining"] == 0
        assert res4["exhausted"] is True
        assert "Demo quota reached for today" in res4["message"]
        assert "Ollama" in res4["message"]

        # Status check now shows exhausted
        status_after = daily_quota.get_quota_status()
        assert status_after["exhausted"] is True
        assert status_after["remaining"] == 0


def test_quota_date_rollover(tmp_path):
    """When the calendar date rolls over, quota counter resets to 0."""
    fake_quota_file = tmp_path / "quota.json"
    # Write yesterday's exhausted quota
    fake_quota_file.write_text(json.dumps({"date": "2020-01-01", "count": 100}), encoding="utf-8")

    with patch.object(daily_quota, "DEMO_MODE", True), \
         patch.object(daily_quota, "DEMO_DAILY_CAP", 5), \
         patch.object(daily_quota, "QUOTA_FILE_PATH", fake_quota_file):

        status = daily_quota.get_quota_status()
        # Today is not 2020-01-01, so count should reset
        assert status["count"] == 0
        assert status["remaining"] == 5
        assert status["exhausted"] is False

        # Should allow consumption
        res = daily_quota.consume_quota()
        assert res["allowed"] is True
        assert res["count"] == 1

