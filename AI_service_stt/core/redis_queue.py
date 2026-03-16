from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

try:
    import redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None

    class RedisError(Exception):
        pass


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STT_QUEUE_KEY = os.getenv("STT_REDIS_QUEUE_KEY", "stt:queue")
STT_TTL = int(os.getenv("STT_REDIS_TTL", "86400"))

_redis_client = None


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"JSON serialization not supported for type={type(value)!r}")


def _get_client():
    global _redis_client

    if redis is None:
        return None

    if _redis_client is None:
        try:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            _redis_client = client
        except Exception as exc:  # pragma: no cover - connection fallback
            print(f"[REDIS][stt] unavailable: {exc}")
            _redis_client = False

    return _redis_client or None


def is_redis_available() -> bool:
    return _get_client() is not None


def set_stt_status(stt_id: str, payload: dict[str, Any]) -> None:
    client = _get_client()
    if not client:
        return

    try:
        client.set(
            f"stt:job:{stt_id}",
            json.dumps(payload, ensure_ascii=False, default=_json_default),
            ex=STT_TTL,
        )
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][stt] status set failed stt_id={stt_id}: {exc}")


def enqueue_stt_job(stt_id: str, file_path: str) -> bool:
    client = _get_client()
    if not client:
        return False

    payload = {"stt_id": stt_id, "file_path": file_path}

    try:
        client.rpush(STT_QUEUE_KEY, json.dumps(payload, ensure_ascii=False))
        return True
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][stt] enqueue failed stt_id={stt_id}: {exc}")
        return False


def dequeue_stt_job(timeout: int = 1) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    try:
        item = client.blpop(STT_QUEUE_KEY, timeout=timeout)
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][stt] dequeue failed: {exc}")
        return None

    if not item:
        return None

    _, raw = item
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

