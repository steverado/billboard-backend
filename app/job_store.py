# app/job_store.py
import json
import redis
from typing import Any, Dict, Optional
from .config import REDIS_URL

_redis = redis.from_url(REDIS_URL)


class JobStore:
    _prefix = "job:"

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        raw = _redis.get(self._key(job_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, job_id: str, data: Dict[str, Any]) -> None:
        _redis.set(self._key(job_id), json.dumps(data))

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        data = self.get(job_id) or {"job_id": job_id}
        data.update(patch)
        self.set(job_id, data)


job_store = JobStore()
