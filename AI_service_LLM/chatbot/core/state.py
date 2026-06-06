# AI_service_LLM/chatbot/core/state.py

from __future__ import annotations

from typing import TypedDict, Literal, List, Dict, Any, Optional


class Message(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str
    # 에이전트 이름, q_score, sources 등 메타데이터
    meta: Dict[str, Any]


class ChatState(TypedDict, total=False):
    """
    LangGraph 에이전트들 사이에서 공유되는 상태 구조.
    기본적으로 messages 배열과 user_id만 있어도 동작한다.
    """
    # ===== 공통 메타 =====
    user_id: Optional[str]
    # 나중에 세션을 재사용하고 싶다면 session_id도 추가 가능
    session_id: Optional[str]

    # 대화 메시지 히스토리
    messages: List[Message]

    # ===== 라우팅 / 응답 =====
    # 슈퍼바이저에서 결정한 라우트(에이전트 이름 등)
    route: Optional[str]

    # 최종 LLM 응답 텍스트 (한 턴 기준)
    answer: Optional[str]

    # ===== 출처 정보 =====
    # 이번 턴에서 사용된 출처 리스트
    # 예: [{"id": "...", "collection": "disease", "title": "...", "url": "...", "score": 0.87}, ...]
    sources: List[Dict[str, Any]]

    # ===== 에이전트 수집 정보 =====
    # 각 에이전트가 수집한 정보(Context)를 리스트로 축적.
    # Final Generator가 이 정보를 종합하여 최종 답변을 생성한다.
    # 예: [{"agent": "drug_agent", "context": "...", "sources": [...], "meta": {...}}, ...]
    context_list: List[Dict[str, Any]]
