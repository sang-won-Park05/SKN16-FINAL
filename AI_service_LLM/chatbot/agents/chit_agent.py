# AI_service_LLM/chatbot/agents/chit_agent.py

from __future__ import annotations

from ..core.state import ChatState
from ..core.tracing import traceable
from ..core.prompts import CHIT_SYSTEM_PROMPT
from ..core.llm import call_llm


@traceable(name="chit_agent")
def run(state: ChatState) -> ChatState:
    """
    비의료 / 일반적인 대화(잡담, 공부, 개발 질문 등)를 처리하는 에이전트.
    RAG 없이 LLM만 사용한다.
    """
    user_message = state["messages"][-1]["content"]

    answer = call_llm(
        system_prompt=CHIT_SYSTEM_PROMPT,
        user_message=user_message,
        context=None,
    )

    if "context_list" not in state:
        state["context_list"] = []

    state["context_list"].append(
        {
            "agent": "chit_agent",
            "context": answer,
            "sources": [],
            "meta": {},
        }
    )

    return state
