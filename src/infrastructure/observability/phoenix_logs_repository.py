import os
from datetime import datetime, timezone, timedelta
from typing import Any

import requests


class PhoenixLogsRepository:
    """
    Reads spans from Phoenix REST API so downstream quality analysis can use
    actual logged input/output pairs.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("PHOENIX_API_URL", "http://localhost:16006").rstrip("/")
        self.timeout_seconds = float(os.getenv("PHOENIX_API_TIMEOUT_SEC", "3.0"))
        self.project_identifier = os.getenv("PHOENIX_PROJECT_IDENTIFIER", "").strip()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params or {},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _resolve_project_identifier(self) -> str | None:
        if self.project_identifier:
            return self.project_identifier
        try:
            payload = self._get_json("/v1/projects", params={"limit": 1})
            rows = payload.get("data", [])
            if isinstance(rows, list) and rows:
                first = rows[0] if isinstance(rows[0], dict) else {}
                pid = str(first.get("id", "")).strip()
                if pid:
                    self.project_identifier = pid
                    return pid
        except Exception as e:
            print(f"[WARNING] Failed to resolve Phoenix project identifier: {e}")
        return None

    @staticmethod
    def _default_start_time() -> str:
        return (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

    def get_spans_by_request_ids(
        self,
        request_ids: set[str],
        start_time: str | None = None,
        limit: int = 300,
        max_pages: int = 6,
    ) -> dict[str, dict[str, Any]]:
        if not request_ids:
            return {}
        project_id = self._resolve_project_identifier()
        if not project_id:
            return {}

        found: dict[str, dict[str, Any]] = {}
        cursor: str | None = None
        start_time = start_time or self._default_start_time()
        safe_limit = max(1, min(limit, 1000))

        for _ in range(max_pages):
            params: dict[str, Any] = {
                "limit": safe_limit,
                "start_time": start_time,
                "name": "yaguarete.chat_completion",
            }
            if cursor:
                params["cursor"] = cursor
            try:
                payload = self._get_json(f"/v1/projects/{project_id}/spans", params=params)
            except Exception as e:
                print(f"[WARNING] Failed to fetch Phoenix spans: {e}")
                break

            rows = payload.get("data", [])
            if not isinstance(rows, list) or not rows:
                break

            for row in rows:
                if not isinstance(row, dict):
                    continue
                attributes = row.get("attributes", {})
                if not isinstance(attributes, dict):
                    continue
                request_id = str(attributes.get("yaguarete.request_id", "")).strip()
                if not request_id or request_id not in request_ids:
                    continue
                found[request_id] = row

            if len(found) == len(request_ids):
                break

            next_cursor = payload.get("next_cursor")
            cursor = str(next_cursor).strip() if next_cursor is not None else None
            if not cursor:
                break

        return found
