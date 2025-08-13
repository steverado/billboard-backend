# app/job_store.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# Prefer config if present, but don't crash if import side-effects raise
try:
    from .config import REDIS_URL as CONFIG_REDIS_URL  # optional
except Exception:  # pragma: no cover
    CONFIG_REDIS_URL = None

REDIS_URL = (CONFIG_REDIS_URL or os.getenv("REDIS_URL") or "").strip()
JOB_STORE_PREFIX = os.getenv("JOB_STORE_PREFIX", "job:")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))  # 24h default


class _MemoryStore:
    """In-process fallback so imports never fail."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        v = self._data.get(key)
        return v.copy() if isinstance(v, dict) else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._data[key] = value.copy()

    def expire(self, key: str, _seconds: int) -> None:  # no-op in memory
        return


class _RedisStore:
    """Thin wrapper to avoid hard dependency at import time."""

    def __init__(self, url: str):
        # Import lazily so missing redis package doesn't hard-crash at import
        from redis import Redis  # type: ignore

        # decode_responses=True gives us str instead of bytes
        self.r = Redis.from_url(url, decode_responses=True)
        # Validate connection early but don't explode the whole app if it fails
        try:
            self.r.ping()
        except Exception:
            # We'll still allow object creation; operations may fail and be caught
            pass

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self.r.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.r.set(key, json.dumps(value))

    def expire(self, key: str, seconds: int) -> None:
        try:
            self.r.expire(key, seconds)
        except Exception:
            pass  # non-fatal


class JobStore:
    """Public API compatible with your previous version."""

    _prefix = JOB_STORE_PREFIX

    def __init__(self):
        self._backend = self._select_backend()

    def _select_backend(self):
        if REDIS_URL:
            try:
                return _RedisStore(REDIS_URL)
            except Exception:
                # Any issue (module missing, bad URL, network) -> memory fallback
                return _MemoryStore()
        return _MemoryStore()

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get(self._key(job_id))

    def set(self, job_id: str, data: Dict[str, Any]) -> None:
        k = self._key(job_id)
        self._backend.set(k, data)
        # Best-effort TTL to avoid unbounded growth in Redis
        self._backend.expire(k, JOB_TTL_SECONDS)

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        data = self.get(job_id) or {"job_id": job_id}
        data.update(patch)
        self.set(job_id, data)


# Export the singleton used throughout the app
job_store = JobStore()
