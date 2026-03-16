from __future__ import annotations

import hashlib
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
DEFAULT_TTL = int(os.getenv("REDIS_DEFAULT_TTL", "86400"))
STT_TTL = int(os.getenv("STT_REDIS_TTL", str(DEFAULT_TTL)))
OCR_TTL = int(os.getenv("OCR_REDIS_TTL", str(DEFAULT_TTL)))
OCR_PARSE_TTL = int(os.getenv("OCR_PARSE_REDIS_TTL", str(DEFAULT_TTL)))

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
            print(f"[REDIS][backend] unavailable: {exc}")
            _redis_client = False

    return _redis_client or None


def _set_json(key: str, payload: dict[str, Any], ttl: int) -> None:
    client = _get_client()
    if not client:
        return

    try:
        client.set(key, json.dumps(payload, ensure_ascii=False, default=_json_default), ex=ttl)
    except RedisError as exc:  # pragma: no cover - network/runtime fallback
        print(f"[REDIS][backend] set failed key={key}: {exc}")


def _get_json(key: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    try:
        raw = client.get(key)
    except RedisError as exc:  # pragma: no cover - network/runtime fallback
        print(f"[REDIS][backend] get failed key={key}: {exc}")
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _delete_keys(*keys: str) -> None:
    client = _get_client()
    if not client or not keys:
        return

    try:
        client.delete(*keys)
    except RedisError as exc:  # pragma: no cover - network/runtime fallback
        print(f"[REDIS][backend] delete failed keys={keys}: {exc}")


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def set_stt_status(stt_id: str, payload: dict[str, Any]) -> None:
    _set_json(f"stt:job:{stt_id}", payload, STT_TTL)


def get_stt_status(stt_id: str) -> dict[str, Any] | None:
    return _get_json(f"stt:job:{stt_id}")


def set_ocr_job_status(ocr_id: int, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:job:{ocr_id}", payload, OCR_TTL)


def get_ocr_result(source_type: str, file_hash: str) -> dict[str, Any] | None:
    return _get_json(f"ocr:result:{source_type}:{file_hash}")


def set_ocr_result(source_type: str, file_hash: str, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:result:{source_type}:{file_hash}", payload, OCR_TTL)


def get_ocr_parse(kind: str, text: str) -> dict[str, Any] | None:
    return _get_json(f"ocr:parse:{kind}:{_hash_value(text.strip())}")


def set_ocr_parse(kind: str, text: str, payload: dict[str, Any]) -> None:
    _set_json(f"ocr:parse:{kind}:{_hash_value(text.strip())}", payload, OCR_PARSE_TTL)


def delete_ocr_job_status(ocr_id: int) -> None:
    _delete_keys(f"ocr:job:{ocr_id}")

