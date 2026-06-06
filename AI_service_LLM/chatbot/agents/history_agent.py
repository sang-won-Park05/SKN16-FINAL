# AI_service_LLM/chatbot/agents/history_agent.py

from __future__ import annotations

from typing import List, Dict

from ..core.state import ChatState
from ..core.tracing import traceable
from ..core.chat_repository import get_recent_logs


@traceable(name="history_agent")
def run(state: ChatState) -> ChatState:
    """
    사용자의 '과거 챗봇 대화 기록(chat_log / chat_session)'을 조회하여
    context_list 에 추가하는 에이전트.
    """
    user_id = state.get("user_id")

    if "context_list" not in state:
        state["context_list"] = []

    # 1) 최근 대화 기록 N개 조회
    logs: List[Dict] = get_recent_logs(user_id=user_id, limit=20)

    if not logs:
        state["context_list"].append(
            {
                "agent": "history_agent",
                "context": "현재 저장된 이전 대화 기록이 없어서, 과거 대화를 참고할 수 없습니다.",
                "sources": [],
                "meta": {"log_count": 0},
            }
        )
        return state

    # 2) context 문자열로 변환
    history_lines: List[str] = []
    for log in logs:
        created_at = log.get("created_at", "")
        # datetime 객체일 수도 있으니 문자열로 캐스팅
        if hasattr(created_at, "isoformat"):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = str(created_at)

        session_id = log.get("session_id", "")
        role = log.get("role", "user")
        content = log.get("content", "")

        # 각 메시지를 한 줄씩 기록
        history_lines.append(
            f"[세션 {session_id} | {created_at_str} | {role}]\n"
            f"{role}: {content}"
        )

    history_context = "\n\n".join(history_lines)

    # 3) state.context_list 에 append
    state["context_list"].append(
        {
            "agent": "history_agent",
            "context": history_context,
            "sources": [],
            "meta": {"log_count": len(logs)},
        }
    )

    return state
