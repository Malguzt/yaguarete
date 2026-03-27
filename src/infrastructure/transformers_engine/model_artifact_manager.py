import threading
from typing import Dict
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from infrastructure.observability.metrics import (
    MODEL_DOWNLOAD_DOWNLOADED_BYTES,
    MODEL_DOWNLOAD_EVENTS_TOTAL,
    MODEL_DOWNLOAD_FILES_COMPLETED,
    MODEL_DOWNLOAD_FILES_TOTAL,
    MODEL_DOWNLOAD_IN_PROGRESS,
    MODEL_DOWNLOAD_PROGRESS_PERCENT,
    MODEL_DOWNLOAD_STATUS_INFO,
    MODEL_DOWNLOAD_TOTAL_BYTES,
)


class ModelArtifactManager:
    """Single responsibility: ensure model artifacts are available locally and emit progress metrics."""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self._download_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    def _get_download_lock(self, model_id: str) -> threading.Lock:
        with self._lock:
            if model_id not in self._download_locks:
                self._download_locks[model_id] = threading.Lock()
            return self._download_locks[model_id]

    def _set_download_progress(
        self,
        model_id: str,
        files_completed: int,
        files_total: int,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        files_total_safe = max(files_total, 0)
        total_bytes_safe = max(total_bytes, 0)
        progress = 0.0
        if total_bytes_safe > 0:
            progress = min(100.0, (downloaded_bytes / total_bytes_safe) * 100.0)
        elif files_total_safe > 0:
            progress = min(100.0, (files_completed / files_total_safe) * 100.0)

        MODEL_DOWNLOAD_FILES_TOTAL.labels(model_id=model_id, node=self.node_name).set(files_total_safe)
        MODEL_DOWNLOAD_FILES_COMPLETED.labels(model_id=model_id, node=self.node_name).set(files_completed)
        MODEL_DOWNLOAD_TOTAL_BYTES.labels(model_id=model_id, node=self.node_name).set(total_bytes_safe)
        MODEL_DOWNLOAD_DOWNLOADED_BYTES.labels(model_id=model_id, node=self.node_name).set(downloaded_bytes)
        MODEL_DOWNLOAD_PROGRESS_PERCENT.labels(model_id=model_id, node=self.node_name).set(progress)

    def _set_download_status(self, model_id: str, current_status: str) -> None:
        for status in ("idle", "in_progress", "success", "error"):
            MODEL_DOWNLOAD_STATUS_INFO.labels(
                model_id=model_id,
                node=self.node_name,
                status=status,
            ).set(1 if status == current_status else 0)

    def try_local_fallback(self, model_id: str) -> bool:
        try:
            snapshot_download(
                repo_id=model_id,
                local_files_only=True,
                resume_download=True,
            )
            self._set_download_progress(model_id, files_completed=1, files_total=1, downloaded_bytes=1, total_bytes=1)
            self._set_download_status(model_id, "success")
            MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="fallback_local").inc()
            return True
        except Exception:
            self._set_download_status(model_id, "error")
            return False

    def ensure_local_artifacts(self, model_id: str) -> None:
        download_lock = self._get_download_lock(model_id)
        with download_lock:
            # Fast path: already local.
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,
                    resume_download=True,
                )
                self._set_download_progress(model_id, files_completed=1, files_total=1, downloaded_bytes=1, total_bytes=1)
                MODEL_DOWNLOAD_IN_PROGRESS.labels(model_id=model_id, node=self.node_name).set(0)
                self._set_download_status(model_id, "success")
                MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="cache_hit").inc()
                return
            except Exception:
                pass

            MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="started").inc()
            MODEL_DOWNLOAD_IN_PROGRESS.labels(model_id=model_id, node=self.node_name).set(1)
            self._set_download_status(model_id, "in_progress")
            self._set_download_progress(model_id, files_completed=0, files_total=0, downloaded_bytes=0, total_bytes=0)

            api = HfApi()
            repo_files = []
            try:
                try:
                    model_info = api.model_info(repo_id=model_id, files_metadata=True)
                    repo_files = [f for f in (model_info.siblings or []) if getattr(f, "rfilename", None)]
                except Exception as metadata_err:
                    print(f"[WARNING] Could not fetch file metadata for {model_id}: {metadata_err}")

                if not repo_files:
                    snapshot_download(
                        repo_id=model_id,
                        local_files_only=False,
                        resume_download=True,
                    )
                    self._set_download_progress(model_id, files_completed=1, files_total=1, downloaded_bytes=1, total_bytes=1)
                    MODEL_DOWNLOAD_IN_PROGRESS.labels(model_id=model_id, node=self.node_name).set(0)
                    self._set_download_status(model_id, "success")
                    MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="success").inc()
                    return

                total_files = len(repo_files)
                total_bytes = sum(int(getattr(repo_file, "size", 0) or 0) for repo_file in repo_files)
                completed_files = 0
                downloaded_bytes = 0
                self._set_download_progress(
                    model_id=model_id,
                    files_completed=completed_files,
                    files_total=total_files,
                    downloaded_bytes=downloaded_bytes,
                    total_bytes=total_bytes,
                )

                for repo_file in repo_files:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=repo_file.rfilename,
                        local_files_only=False,
                        resume_download=True,
                    )
                    completed_files += 1
                    downloaded_bytes += int(getattr(repo_file, "size", 0) or 0)
                    self._set_download_progress(
                        model_id=model_id,
                        files_completed=completed_files,
                        files_total=total_files,
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes,
                    )

                MODEL_DOWNLOAD_IN_PROGRESS.labels(model_id=model_id, node=self.node_name).set(0)
                self._set_download_status(model_id, "success")
                MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="success").inc()
            except Exception:
                MODEL_DOWNLOAD_IN_PROGRESS.labels(model_id=model_id, node=self.node_name).set(0)
                self._set_download_status(model_id, "error")
                MODEL_DOWNLOAD_EVENTS_TOTAL.labels(model_id=model_id, node=self.node_name, status="error").inc()
                raise
