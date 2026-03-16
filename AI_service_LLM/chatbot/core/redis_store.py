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
LLM_REDIS_TTL = int(os.getenv("LLM_REDIS_TTL", "86400"))
LLM_ANALYSIS_REDIS_TTL = int(os.getenv("LLM_ANALYSIS_REDIS_TTL", "3600"))
LLM_SESSION_MESSAGE_LIMIT = int(os.getenv("LLM_SESSION_MESSAGE_LIMIT", "40"))
LLM_RECENT_LOG_LIMIT = int(os.getenv("LLM_RECENT_LOG_LIMIT", "100"))

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
            print(f"[REDIS][llm] unavailable: {exc}")
            _redis_client = False

    return _redis_client or None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_key(user_id: int, session_id: int, query: str) -> str:
    digest = _hash_text(f"{session_id}:{query.strip()}")
    return f"llm:query:{user_id}:{digest}"


def _analysis_key(user_id: int, context: str) -> str:
    digest = _hash_text(context.strip())
    return f"llm:analysis:{user_id}:{digest}"


def _session_messages_key(session_id: int) -> str:
    return f"llm:session:{session_id}:messages"


def _recent_logs_key(user_id: int) -> str:
    return f"llm:user:{user_id}:recent_logs"


def _user_sessions_key(user_id: int) -> str:
    return f"llm:user:{user_id}:sessions"


def _decode_json_items(items: list[str]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for item in items:
        try:
            value = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append(value)
    return decoded


def _delete_keys(*keys: str) -> None:
    client = _get_client()
    if not client or not keys:
        return

    try:
        client.delete(*keys)
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] delete failed keys={keys}: {exc}")


def get_cached_query_response(
    *,
    user_id: int,
    session_id: int,
    query: str,
) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    try:
        raw = client.get(_query_key(user_id, session_id, query))
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] query cache get failed: {exc}")
        return None

    if not raw:
        return None

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, dict) else None


def set_cached_query_response(
    *,
    user_id: int,
    session_id: int,
    query: str,
    answer: str,
    sources: list[dict[str, Any]] | None,
) -> None:
    client = _get_client()
    if not client:
        return

    payload = {
        "answer": answer,
        "sources": sources or [],
    }

    try:
        client.set(
            _query_key(user_id, session_id, query),
            json.dumps(payload, ensure_ascii=False, default=_json_default),
            ex=LLM_REDIS_TTL,
        )
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] query cache set failed: {exc}")


def get_cached_analysis(*, user_id: int, context: str) -> str | None:
    client = _get_client()
    if not client:
        return None

    try:
        raw = client.get(_analysis_key(user_id, context))
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] analysis cache get failed: {exc}")
        return None

    if not raw:
        return None

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(value, dict):
        return value.get("analysis")
    return None


def set_cached_analysis(*, user_id: int, context: str, analysis: str) -> None:
    client = _get_client()
    if not client:
        return

    try:
        client.set(
            _analysis_key(user_id, context),
            json.dumps({"analysis": analysis}, ensure_ascii=False),
            ex=LLM_ANALYSIS_REDIS_TTL,
        )
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] analysis cache set failed: {exc}")


def set_session_messages(
    *,
    session_id: int,
    user_id: int,
    messages: list[dict[str, Any]],
) -> None:
    client = _get_client()
    if not client:
        return

    key = _session_messages_key(session_id)
    user_sessions_key = _user_sessions_key(user_id)

    try:
        pipe = client.pipeline()
        pipe.delete(key)
        if messages:
            pipe.rpush(
                key,
                *[
                    json.dumps(message, ensure_ascii=False, default=_json_default)
                    for message in messages
                ],
            )
        pipe.ltrim(key, -LLM_SESSION_MESSAGE_LIMIT, -1)
        pipe.expire(key, LLM_REDIS_TTL)
        pipe.sadd(user_sessions_key, session_id)
        pipe.expire(user_sessions_key, LLM_REDIS_TTL)
        pipe.execute()
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] set session messages failed session_id={session_id}: {exc}")


def append_session_messages(
    *,
    session_id: int,
    user_id: int,
    messages: list[dict[str, Any]],
) -> None:
    client = _get_client()
    if not client or not messages:
        return

    key = _session_messages_key(session_id)
    user_sessions_key = _user_sessions_key(user_id)

    try:
        pipe = client.pipeline()
        pipe.rpush(
            key,
            *[
                json.dumps(message, ensure_ascii=False, default=_json_default)
                for message in messages
            ],
        )
        pipe.ltrim(key, -LLM_SESSION_MESSAGE_LIMIT, -1)
        pipe.expire(key, LLM_REDIS_TTL)
        pipe.sadd(user_sessions_key, session_id)
        pipe.expire(user_sessions_key, LLM_REDIS_TTL)
        pipe.execute()
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] append session messages failed session_id={session_id}: {exc}")


def get_cached_session_messages(session_id: int) -> list[dict[str, Any]] | None:
    client = _get_client()
    if not client:
        return None

    try:
        items = client.lrange(_session_messages_key(session_id), 0, -1)
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] get session messages failed session_id={session_id}: {exc}")
        return None

    if not items:
        return None

    return _decode_json_items(items)


def set_recent_logs(user_id: int, messages: list[dict[str, Any]]) -> None:
    client = _get_client()
    if not client:
        return

    key = _recent_logs_key(user_id)
    try:
        pipe = client.pipeline()
        pipe.delete(key)
        if messages:
            pipe.rpush(
                key,
                *[
                    json.dumps(message, ensure_ascii=False, default=_json_default)
                    for message in messages
                ],
            )
        pipe.ltrim(key, 0, LLM_RECENT_LOG_LIMIT - 1)
        pipe.expire(key, LLM_REDIS_TTL)
        pipe.execute()
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] set recent logs failed user_id={user_id}: {exc}")


def append_recent_logs(
    *,
    user_id: int,
    session_id: int,
    messages: list[dict[str, Any]],
) -> None:
    client = _get_client()
    if not client or not messages:
        return

    key = _recent_logs_key(user_id)
    try:
        pipe = client.pipeline()
        for message in messages:
            entry = {"session_id": session_id, **message}
            pipe.lpush(key, json.dumps(entry, ensure_ascii=False, default=_json_default))
        pipe.ltrim(key, 0, LLM_RECENT_LOG_LIMIT - 1)
        pipe.expire(key, LLM_REDIS_TTL)
        pipe.execute()
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] append recent logs failed user_id={user_id}: {exc}")


def get_cached_recent_logs(user_id: int, limit: int) -> list[dict[str, Any]] | None:
    client = _get_client()
    if not client:
        return None

    try:
        items = client.lrange(_recent_logs_key(user_id), 0, max(limit - 1, 0))
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] get recent logs failed user_id={user_id}: {exc}")
        return None

    if not items:
        return None

    return _decode_json_items(items)


def clear_session_cache(session_id: int, user_id: int | None = None) -> None:
    client = _get_client()
    if not client:
        return

    key = _session_messages_key(session_id)
    try:
        pipe = client.pipeline()
        pipe.delete(key)
        if user_id is not None:
            pipe.srem(_user_sessions_key(user_id), session_id)
        pipe.execute()
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] clear session cache failed session_id={session_id}: {exc}")


def clear_user_cache(user_id: int) -> None:
    client = _get_client()
    if not client:
        return

    sessions_key = _user_sessions_key(user_id)

    try:
        session_ids = [int(value) for value in client.smembers(sessions_key)]
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] read user sessions failed user_id={user_id}: {exc}")
        session_ids = []

    keys = [_recent_logs_key(user_id), sessions_key]
    keys.extend(_session_messages_key(session_id) for session_id in session_ids)
    _delete_keys(*keys)


def clear_all_llm_cache() -> None:
    client = _get_client()
    if not client:
        return

    try:
        keys = client.keys("llm:*")
    except RedisError as exc:  # pragma: no cover - runtime fallback
        print(f"[REDIS][llm] scan failed: {exc}")
        return

    if keys:
        _delete_keys(*keys)
