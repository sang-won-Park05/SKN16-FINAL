# AI_service_LLM/chatbot/agents/web_agent.py

from __future__ import annotations

from typing import List, Dict, Any

from ..core.state import ChatState
from ..core.tracing import traceable

try:
    from ..core.web_search import search_web  # type: ignore
except ImportError:
    def search_web(query: str, top_k: int = 5) -> List[Dict[str, Any]]:  # type: ignore
        # 임시 더미 구현
        return []


@traceable(name="web_agent")
def run(state: ChatState) -> ChatState:
    """
    신뢰할 수 있는 웹 검색 결과를 조회하여
    context_list 에 추가하는 에이전트.
    """
    user_message = state["messages"][-1]["content"]

    if "context_list" not in state:
        state["context_list"] = []

    # ------------------------------------------------
    # 1) 웹 검색
    # ------------------------------------------------
    web_results: List[Dict[str, Any]] = search_web(user_message, top_k=5)

    # ------------------------------------------------
    # 2) 컨텍스트 & 출처 리스트 구성
    # ------------------------------------------------
    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, r in enumerate(web_results):
        title = r.get("title") or "웹 검색 결과"
        url = r.get("url")
        snippet = r.get("snippet") or ""
        score = r.get("score")

        context_parts.append(f"{title}\n{snippet}")

        sources.append(
            {
                "id": r.get("id") or url or f"web_result_{idx}",
                "collection": "web",
                "title": title,
                "url": url,
                "score": float(score) if isinstance(score, (float, int)) else None,
            }
        )

    if context_parts:
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = "요청하신 내용에 대해 신뢰할 수 있는 웹 검색 결과를 찾지 못했습니다."

    # ------------------------------------------------
    # 3) state.context_list 에 append
    # ------------------------------------------------
    state["context_list"].append(
        {
            "agent": "web_agent",
            "context": context_text,
            "sources": sources,
            "meta": {"result_count": len(web_results)},
        }
    )

    return state
