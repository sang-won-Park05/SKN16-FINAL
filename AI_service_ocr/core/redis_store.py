from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from core.config import OCR_PARSE_REDIS_TTL, OCR_REDIS_TTL, REDIS_URL

try:
    import redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None

    class RedisError(Exception):
        pass


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
            print(f"[REDIS][ocr] unavailable: {exc}")
            _redis_client = False

    return _redis_client or None


def _set_json(key: str, payload: dict[str, Any], ttl: int) -> None:
    client = _get_client()
    if not client:
        return

    try:
        client.set(key, json.dumps(payload, ensure_ascii=False, default=_json_default), ex=ttl)
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][ocr] set failed key={key}: {exc}")


def _get_json(key: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    try:
        raw = client.get(key)
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][ocr] get failed key={key}: {exc}")
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_ocr_result(source_type: str, file_hash: str) -> dict[str, Any] | None:
    return _get_json(f"ocr:result:{source_type}:{file_hash}")


def set_ocr_result(source_type: str, file_hash: str, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:result:{source_type}:{file_hash}", payload, OCR_REDIS_TTL)


def set_ocr_job_status(ocr_id: int, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:job:{ocr_id}", payload, OCR_REDIS_TTL)


def get_parse_result(kind: str, text: str) -> dict[str, Any] | None:
    return _get_json(f"ocr:parse:{kind}:{hash_text(text.strip())}")


def set_parse_result(kind: str, text: str, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:parse:{kind}:{hash_text(text.strip())}", payload, OCR_PARSE_REDIS_TTL)
