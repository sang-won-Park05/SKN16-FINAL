# AI_service_LLM/chatbot/api/router.py

from __future__ import annotations

import os
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ..graph import chatbot_graph
from ..core.state import ChatState
from ..core.chat_repository import (
    upsert_session_with_log,
    list_sessions,
    get_session_messages,
    delete_all_sessions,
    delete_session,
)
from ..core.redis_store import (
    get_cached_query_response,
    set_cached_query_response,
)

router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
)

DEFAULT_USER_ID = int(
    os.getenv("LLM_DEFAULT_USER_ID")
    or os.getenv("DEFAULT_USER_ID")
    or "1"
)


def _resolve_user_id(user_id: Optional[int]) -> int:
    """
    user_id가 없으면 환경변수 또는 기본값을 사용한다.
    """
    return user_id if user_id is not None else DEFAULT_USER_ID


# =========================
# Pydantic Models
# =========================

class ChatSource(BaseModel):
    """
    LangGraph 에이전트가 생성한 '출처' 정보.
    - disease / drug: Chroma 문서
    - web: Tavily 웹 검색 결과
    """
    id: str
    collection: str                    # disease / drug / web ...
    title: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None


class ChatQueryRequest(BaseModel):
    """
    POST /chatbot/query 요청 바디

    - session_id: 0 이면 새 대화방 생성, 0이 아니면 해당 세션에 이어서 질문
    - query     : 사용자 질문
    """
    session_id: int = Field(0, ge=0, description="기존 세션 ID (0이면 새 세션)")
    query: str = Field(..., description="사용자 질문 텍스트")


class ChatQueryResponse(BaseModel):
    """
    POST /chatbot/query 응답 바디

    - session_id: upsert 된 대화방 ID
    - answer    : 챗봇 최종 답변
    - sources   : 답변에 사용된 출처(벡터DB/웹) 메타 정보
    """
    session_id: int
    answer: str
    sources: List[ChatSource] = []


class SessionItem(BaseModel):
    """
    GET /chatbot/sessions 한 줄 정보
    """
    session_id: int
    title: str
    created_at: datetime   # 🔥 ISO8601로 직렬화됨


class SessionsResponse(BaseModel):
    """
    GET /chatbot/sessions 응답 바디
    """
    sessions: List[SessionItem]


class SessionMessage(BaseModel):
    """
    GET /chatbot/sessions/{session_id} 메시지 한 줄
    """
    role: str
    content: str
    created_at: datetime   # 🔥 ISO8601로 직렬화됨


class SessionDetailResponse(BaseModel):
    """
    GET /chatbot/sessions/{session_id} 응답 바디
    """
    session_id: int
    messages: List[SessionMessage]


class MessageResponse(BaseModel):
    """
    DELETE 응답 바디 공통
    """
    message: str


# =========================
# Routes
# =========================


@router.post("/query", response_model=ChatQueryResponse)
async def chatbot_query(payload: ChatQueryRequest) -> ChatQueryResponse:
    """
    메인 챗봇 엔드포인트.

    흐름:
    1) ChatState 구성 (user_id / session_id / messages)
    2) LangGraph(chatbot_graph) 실행
    3) result에서 answer / sources / messages 꺼내기
    4) upsert_session_with_log 로 세션 + 로그 저장
    5) session_id / answer / sources 를 응답
    """
    try:
        # 🔥 현재 LLM 서비스는 고정 user_id 사용 (인증 연동 전)
        user_id = _resolve_user_id(None)

        # ------------------------------
        # 1) LangGraph state 구성
        # ------------------------------
        state: ChatState = {
            "user_id": str(user_id),
            "messages": [
                {
                    "role": "user",
                    "content": payload.query,
                    "meta": {},
                }
            ],
        }

        # 기존 세션이 있다면 state에 힌트로 넣어준다.
        if payload.session_id:
            state["session_id"] = str(payload.session_id)

        cached_response = get_cached_query_response(
            user_id=user_id,
            session_id=payload.session_id,
            query=payload.query,
        )
        if cached_response:
            answer_text = cached_response.get("answer") or ""
            sources_raw = cached_response.get("sources") or []
            sources: List[ChatSource] = (
                [ChatSource(**s) for s in sources_raw if isinstance(s, dict)]
                if sources_raw else []
            )
        else:
            # ------------------------------
            # 2) LangGraph 실행
            # ------------------------------
            result: ChatState = chatbot_graph.invoke(state)

            # ------------------------------
            # 3) answer / sources 추출
            # ------------------------------
            answer_text = result.get("answer") or ""

            # safety: answer 비어있으면 마지막 assistant 메시지에서 fallback
            if not answer_text:
                messages = result.get("messages") or []
                if messages and messages[-1].get("role") == "assistant":
                    answer_text = messages[-1].get("content", "")

            if not answer_text:
                answer_text = (
                    "죄송합니다. 현재는 적절한 답변을 생성하지 못했습니다. "
                    "질문을 조금 더 구체적으로 말씀해 주시면 도움이 됩니다."
                )

            sources_raw = result.get("sources") or []
            sources = (
                [ChatSource(**s) for s in sources_raw if isinstance(s, dict)]
                if sources_raw else []
            )
            set_cached_query_response(
                user_id=user_id,
                session_id=payload.session_id,
                query=payload.query,
                answer=answer_text,
                sources=[s.model_dump() for s in sources] if sources else None,
            )

        # ------------------------------
        # 4) DB 저장 (세션 upsert + 로그 저장)
        #   - chat_repository.upsert_session_with_log 시그니처에 맞게 호출
        # ------------------------------
        session_id = upsert_session_with_log(
            session_id=payload.session_id if payload.session_id else None,
            user_id=user_id,
            query=payload.query,
            answer=answer_text,
            sources=[s.model_dump() for s in sources] if sources else None,
        )

        # ------------------------------
        # 5) 응답
        # ------------------------------
        return ChatQueryResponse(
            session_id=int(session_id),
            answer=answer_text,
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"챗봇 처리 중 오류가 발생했습니다: {e}",
        )


@router.get("/sessions", response_model=SessionsResponse)
async def get_sessions(
    user_id: int | None = Query(
        None, ge=1, description="조회할 사용자 ID (없으면 기본값 사용)"
    ),
) -> SessionsResponse:
    """
    모든 챗봇 세션 목록 조회.
    """
    try:
        rows = list_sessions(user_id=_resolve_user_id(user_id))
        items = [
            SessionItem(
                session_id=int(row["session_id"]),
                title=row["title"],
                created_at=row["created_at"],  # 🔥 datetime 그대로 전달
            )
            for row in rows
        ]
        return SessionsResponse(sessions=items)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 목록을 불러오는 중 오류가 발생했습니다: {e}",
        )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: int = Path(..., ge=1, description="세션 ID"),
    user_id: int | None = Query(
        None, ge=1, description="조회할 사용자 ID (없으면 기본값 사용)"
    ),
) -> SessionDetailResponse:
    """
    특정 세션의 전체 메시지 내역 조회.
    """
    try:
        rows = get_session_messages(
            session_id=session_id,
            user_id=_resolve_user_id(user_id),
        )
        if not rows:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        messages = [
            SessionMessage(
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],  # 🔥 datetime 그대로
            )
            for row in rows
        ]
        return SessionDetailResponse(
            session_id=session_id,
            messages=messages,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 내역을 불러오는 중 오류가 발생했습니다: {e}",
        )


@router.delete("/sessions", response_model=MessageResponse)
async def delete_sessions_all(
    user_id: int | None = Query(
        None, ge=1, description="삭제할 사용자 ID (없으면 기본값 사용)"
    ),
    include_all: bool = Query(
        False,
        description="모든 사용자 세션을 삭제 (개발/관리용)",
    ),
) -> MessageResponse:
    """
    모든 세션 + 관련 로그 삭제.
    """
    try:
        target_user_id = None
        if not include_all or user_id is not None:
            target_user_id = _resolve_user_id(user_id)

        delete_all_sessions(
            user_id=target_user_id,
            include_all=include_all,
        )
        return MessageResponse(
            message="모든 챗봇 세션을 삭제했습니다." if include_all else "사용자 세션을 삭제했습니다."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 삭제 중 오류가 발생했습니다: {e}",
        )


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def delete_session_one(
    session_id: int = Path(..., ge=1, description="세션 ID"),
    user_id: int | None = Query(
        None, ge=1, description="삭제할 사용자 ID (없으면 기본값 사용)"
    ),
) -> MessageResponse:
    """
    특정 세션 삭제.
    """
    try:
        deleted = delete_session(
            session_id=session_id,
            user_id=_resolve_user_id(user_id),
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        return MessageResponse(message=f"세션 {session_id}을(를) 삭제했습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 삭제 중 오류가 발생했습니다: {e}",
        )
