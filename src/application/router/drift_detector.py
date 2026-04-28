import os
import time
from typing import Optional

from infrastructure.observability.metrics import (
    ROUTER_MODEL_COOLDOWN,
    ROUTER_MODEL_DRIFT_EVENTS_TOTAL,
    ROUTER_MODEL_DRIFT_SCORE,
)


class DriftDetector:
    """
    Detects short-term degradation vs baseline and applies temporary cooldown.
    """

    def __init__(self, stats_repo) -> None:
        self.stats_repo = stats_repo
        self.enabled = os.getenv("ENABLE_DRIFT_DETECTION", "1").lower() in ("1", "true", "yes")
        self.threshold = float(os.getenv("ROUTER_DRIFT_THRESHOLD", "-0.20"))
        self.cooldown_minutes = max(1, int(os.getenv("ROUTER_DRIFT_COOLDOWN_MINUTES", "15")))
        self.recent_hours = max(1, int(os.getenv("ROUTER_DRIFT_RECENT_HOURS", "2")))
        self.baseline_hours = max(self.recent_hours + 1, int(os.getenv("ROUTER_DRIFT_BASELINE_HOURS", "24")))
        self.min_recent_samples = max(3, int(os.getenv("ROUTER_DRIFT_MIN_RECENT_SAMPLES", "6")))
        self.min_baseline_samples = max(self.min_recent_samples, int(os.getenv("ROUTER_DRIFT_MIN_BASELINE_SAMPLES", "20")))
        self.check_interval_seconds = max(10, int(os.getenv("ROUTER_DRIFT_CHECK_INTERVAL_SEC", "30")))

        self._cooldown_until: dict[str, float] = {}
        self._last_check_at: dict[str, float] = {}

    def _is_cooldown_active(self, model_id: str) -> bool:
        until = self._cooldown_until.get(model_id)
        if until is None:
            return False
        if time.time() >= until:
            self._cooldown_until.pop(model_id, None)
            ROUTER_MODEL_COOLDOWN.labels(model_id=model_id).set(0)
            ROUTER_MODEL_DRIFT_EVENTS_TOTAL.labels(model_id=model_id, event="cooldown_expired").inc()
            return False
        return True

    def _set_cooldown(self, model_id: str, reason: str) -> None:
        until = time.time() + (self.cooldown_minutes * 60.0)
        self._cooldown_until[model_id] = until
        ROUTER_MODEL_COOLDOWN.labels(model_id=model_id).set(1)
        ROUTER_MODEL_DRIFT_EVENTS_TOTAL.labels(model_id=model_id, event=f"cooldown_{reason}").inc()
        print(
            f"[WARNING] Drift cooldown for {model_id}: {self.cooldown_minutes}m "
            f"(reason={reason})"
        )

    def evaluate_model(self, model_id: str) -> Optional[dict[str, float]]:
        if not self.enabled:
            return None

        if self._is_cooldown_active(model_id):
            return {"cooldown": 1.0}

        now = time.time()
        last_check = self._last_check_at.get(model_id, 0.0)
        if (now - last_check) < self.check_interval_seconds:
            return None
        self._last_check_at[model_id] = now

        stats = self.stats_repo.get_model_effectiveness_window_stats(
            model_id=model_id,
            recent_hours=self.recent_hours,
            baseline_hours=self.baseline_hours,
        )
        if not stats:
            return None

        drift_delta = float(stats["drift_delta"])
        ROUTER_MODEL_DRIFT_SCORE.labels(model_id=model_id).set(drift_delta)
        ROUTER_MODEL_COOLDOWN.labels(model_id=model_id).set(0)

        recent_count = int(stats["recent_count"])
        baseline_count = int(stats["baseline_count"])
        if recent_count < self.min_recent_samples or baseline_count < self.min_baseline_samples:
            return stats

        if drift_delta <= self.threshold:
            self._set_cooldown(model_id=model_id, reason="drift")
            stats["cooldown"] = 1.0
        return stats

    def should_skip_model(self, model_id: str) -> bool:
        if not self.enabled:
            return False
        if self._is_cooldown_active(model_id):
            return True
        self.evaluate_model(model_id)
        return self._is_cooldown_active(model_id)
