import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from infrastructure.observability.phoenix_logs_repository import PhoenixLogsRepository
from infrastructure.transformers_engine.model_catalog import ModelCatalog


class PhoenixFeedbackScheduler:
    """
    Background evaluator that waits for inactivity windows and then analyzes
    responses from Phoenix logs to update router quality signals.
    """

    def __init__(self, stats_repo, quality_evaluator) -> None:
        self.stats_repo = stats_repo
        self.quality_evaluator = quality_evaluator
        self.phoenix_repo = PhoenixLogsRepository()

        self.idle_seconds = max(1.0, float(os.getenv("QUALITY_IDLE_SECONDS", "1.0")))
        self.loop_interval_seconds = max(0.2, float(os.getenv("QUALITY_LOOP_INTERVAL_SEC", "0.5")))
        self.phoenix_start_lookback_sec = max(
            60, int(os.getenv("PHOENIX_LOOKBACK_SECONDS", "600"))
        )

        self._pending: dict[str, dict[str, Any]] = {}
        self._last_request_completed_at = 0.0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="PhoenixFeedbackScheduler", daemon=True)
        self._thread.start()
        print(
            "[INFO] Phoenix feedback scheduler started "
            f"(idle={self.idle_seconds:.2f}s, poll={self.loop_interval_seconds:.2f}s)"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def register_completed_request(self, request_id: str, model_id: str, duration_ms: float) -> None:
        with self._lock:
            self._pending[request_id] = {
                "model_id": model_id,
                "duration_ms": duration_ms,
                "registered_at": time.time(),
            }
            self._last_request_completed_at = time.time()

    def _is_idle(self) -> bool:
        with self._lock:
            if self._last_request_completed_at <= 0:
                return False
            return (time.time() - self._last_request_completed_at) >= self.idle_seconds

    def _best_local_model_id(self) -> str | None:
        catalog = ModelCatalog()
        local_generation_models = catalog.get_generation_models(local_only=True)
        if not local_generation_models:
            return None
        # Catalog order already reflects llmfit ranking replacement when enabled.
        return local_generation_models[0].huggingface_id

    @staticmethod
    def _safe_attr(attributes: dict[str, Any], key: str) -> str:
        value = attributes.get(key, "")
        if value is None:
            return ""
        return str(value)

    def _process_pending_batch(self) -> None:
        with self._lock:
            pending_ids = set(self._pending.keys())
        if not pending_ids:
            return

        start_time = (
            datetime.now(timezone.utc) - timedelta(seconds=self.phoenix_start_lookback_sec)
        ).isoformat()
        spans_by_request = self.phoenix_repo.get_spans_by_request_ids(
            request_ids=pending_ids,
            start_time=start_time,
        )
        if not spans_by_request:
            return

        analysis_model_id = self._best_local_model_id()
        for request_id, span in spans_by_request.items():
            attributes = span.get("attributes", {}) if isinstance(span, dict) else {}
            if not isinstance(attributes, dict):
                continue
            prompt = self._safe_attr(attributes, "yaguarete.input")
            response = self._safe_attr(attributes, "yaguarete.output")
            if not prompt or not response:
                continue

            quality_scores = self.quality_evaluator.evaluate_response(
                prompt,
                response,
                analysis_model_id=analysis_model_id,
            )
            self.stats_repo.update_quality_scores(request_id=request_id, **quality_scores)

            pending_payload: dict[str, Any] | None
            with self._lock:
                pending_payload = self._pending.pop(request_id, None)

            if pending_payload:
                from infrastructure.observability.metrics import ROUTER_MODEL_EFFECTIVENESS, ROUTER_AVG_TIME_PER_CHAR

                model_id = str(pending_payload.get("model_id", "unknown"))
                duration_ms = float(pending_payload.get("duration_ms", 0.0))
                combined_eff = (
                    (quality_scores["judge_score"] * 0.5)
                    + (quality_scores["format_score"] * 0.3)
                    + (quality_scores["density_score"] * 0.2)
                )
                ROUTER_MODEL_EFFECTIVENESS.labels(model_id=model_id).set(combined_eff)
                ROUTER_AVG_TIME_PER_CHAR.labels(model_id=model_id).set(duration_ms / max(len(prompt), 1))

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._is_idle():
                    self._process_pending_batch()
            except Exception as e:
                print(f"[WARNING] Phoenix feedback scheduler cycle failed: {e}")
            finally:
                self._stop_event.wait(self.loop_interval_seconds)
