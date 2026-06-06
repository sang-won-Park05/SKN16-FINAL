# AI_service_LLM/chatbot/graph_doc.py

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .core.state import ChatState

"""
실행용이 아니라 '문서/시각화용' 그래프.

- 실제 서비스: chatbot.graph.chatbot_graph 사용
- 문서/그림:   chatbot.graph_doc.doc_graph 사용
"""

def _passthrough(state: ChatState) -> ChatState:
    """그냥 state 그대로 넘기는 더미 노드."""
    return state


# -------------------------------
# 문서용 그래프 빌더
# -------------------------------
builder = StateGraph(ChatState)

# 1) Supervisor / Orchestrator 레벨
builder.add_node("supervisor_planner", _passthrough)

# 2) 에이전트들
builder.add_node("disease_agent", _passthrough)
builder.add_node("drug_agent", _passthrough)
builder.add_node("history_agent", _passthrough)
builder.add_node("db_agent", _passthrough)
builder.add_node("web_agent", _passthrough)
builder.add_node("chit_agent", _passthrough)

# 3) RAG 관련 컴포넌트
builder.add_node("retriever_reranker", _passthrough)

# 4) 최종 답변 생성기 (Final Generator)
builder.add_node("final_generator", _passthrough)

# -------------------------------
# 엔트리 포인트
# -------------------------------
builder.set_entry_point("supervisor_planner")

# -------------------------------
# 엣지 (구조 표현)
# -------------------------------

# supervisor 가 사용자의 질문을 분석해서 각 에이전트로 라우팅
builder.add_edge("supervisor_planner", "disease_agent")
builder.add_edge("supervisor_planner", "drug_agent")
builder.add_edge("supervisor_planner", "history_agent")
builder.add_edge("supervisor_planner", "db_agent")
builder.add_edge("supervisor_planner", "web_agent")
builder.add_edge("supervisor_planner", "chit_agent")

# RAG 사용하는 에이전트들
builder.add_edge("disease_agent", "retriever_reranker")
builder.add_edge("drug_agent", "retriever_reranker")

# RAG 결과 및 정보 수집 완료 후 최종 답변 생성기로
builder.add_edge("retriever_reranker", "final_generator")
builder.add_edge("web_agent", "final_generator")
builder.add_edge("history_agent", "final_generator")
builder.add_edge("db_agent", "final_generator")

# chit_agent는 단독일 땐 END로 가지만, 문서상으로는 최종 생성기로 병합되는 흐름으로 표현
builder.add_edge("chit_agent", "final_generator")

# -------------------------------
# 4) 최종 답변 후 종료
# -------------------------------
builder.add_edge("final_generator", END)

# -------------------------------
# 최종 문서용 그래프
# -------------------------------
doc_graph = builder.compile()
