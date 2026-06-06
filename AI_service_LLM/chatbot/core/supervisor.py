# AI_service_LLM/chatbot/core/supervisor.py

from __future__ import annotations

import json
from typing import List, Literal

from .state import ChatState
from .tracing import traceable
from .llm import call_llm
from .prompts import FINAL_GENERATOR_PROMPT

# 각 에이전트 import
from ..agents import (
    chit_agent,
    db_agent,
    disease_agent,
    drug_agent,
    web_agent,
    history_agent,
)

# =========================================================
# 0) RouteName 타입 정의
# =========================================================

RouteName = Literal["chit", "db", "disease", "drug", "web", "history"]


# =========================================================
# 1) 규칙 기반 1차 라우터: route_supervisor
# =========================================================

def _get_last_user_message(state: ChatState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    return messages[-1].get("content", "") or ""


def route_supervisor(state: ChatState) -> RouteName:
    """
    유저 질문을 보고 적절한 에이전트로 라우팅하는 규칙 기반 1차 Supervisor.
    """

    text = _get_last_user_message(state)
    if not text:
        return "chit"

    t = text.strip()

    # 1) 과거 대화 관련 (history agent)
    history_keywords = [
        "지난번", "예전에", "이전 대화", "지난 대화",
        "전에 했던", "예전에 뭐라고", "지난 기록",
        "예전에 뭐라고 했었지", "물어봤었지",
        "이전 질문", "이전 답변",
    ]
    if any(k in t for k in history_keywords):
        return "history"

    # 2) 개인 의료 DB
    db_keywords = [
        "진료 기록", "진료기록", "진료 내역", "진료내역",
        "처방전", "내 처방", "지난 처방",
        "검사 결과", "검사결과",
        "입원 기록", "입원기록",
        "나의 기록", "내 기록", "나의 진료", "개인 기록",
        "병원 다녀온", "의무기록",
        "건강 분석", "건강분석", "건강 상태", "건강상태",
        "내 건강", "나의 건강", "건강 정보", "건강정보",
        "프로필", "내 프로필", "건강 프로필",
        "BMI", "체질량", "키", "몸무게", "음주", "흡연", "약 정보", "질환 정보", "알러지"
    ]
    if any(k in t for k in db_keywords):
        return "db"

    # 3) 약/영양제/상호작용
    drug_keywords = [
        "약", "약을", "약이", "약은", "정(", "캡슐", "시럽",
        "복용", "복용법", "복용해도", "먹어도", "먹으면",
        "영양제", "비타민", "건강기능식품", "건기식",
        "상호작용", "같이 먹어도", "병용", "같이 먹으면",
        "충돌", "부작용", "반응이", "알약",
        "약국", "처방약", "의약품",
    ]
    if any(k in t for k in drug_keywords):
        return "drug"

    # 4) 질병/증상/진료과
    disease_keywords = [
        "증상", "아픈", "아파요", "통증", "두통", "복통",
        "메스꺼움", "구토", "설사", "변비",
        "열이", "발열", "발진", "기침", "가래",
        "호흡곤란", "숨쉬기", "호흡",
        "진료과", "어느 과", "어떤 과",
        "검사해야", "검사를 받아야",
        "질병", "병명", "병인가요",
    ]
    if any(k in t for k in disease_keywords):
        return "disease"

    # 5) 최신 뉴스 / 새로운 정보 / 특정 연도
    web_keywords = [
        "최신", "최근", "요즘", "업데이트",
        "새로 나온", "신약", "리콜", "뉴스",
        "2023", "2024", "2025", "2026", "2027",
        "최근 연구", "최근 발표",
    ]
    if any(k in t for k in web_keywords):
        return "web"

    # 6) 그 외
    return "chit"


# =========================================================
# 2) LLM 기반 플래너 프롬프트
# =========================================================

SUPERVISOR_PLANNER_SYSTEM_PROMPT = """
너는 '메디노트'의 멀티 에이전트 오케스트레이터이다.

역할:
- 사용자의 질문을 읽고, 아래 6개의 에이전트 중 어떤 것들을 사용할지, 어떤 순서로 호출할지 결정한다.
- 필요한 경우 여러 에이전트를 함께 사용해도 된다.
- 출력은 반드시 JSON 형식으로만 내보낸다. 다른 설명 문장은 절대 포함하지 않는다.

에이전트 종류:
1) "chit"     : 일반 잡담, 비의료 대화, 공부/개발 질문 등
2) "db"       : 사용자의 개인 의료 기록/프로필/과거 진료 기록/처방전/복용 이력 등
3) "disease"  : 질병, 증상, 진료과, 검사 관련 질문
4) "drug"     : 약, 복용법, 약물 상호작용, 영양제/건기식 관련 질문
5) "web"      : 최신 뉴스, 최근 연구, 최신 가이드라인 등 외부 웹 검색이 필요한 경우
6) "history"  : 이전 챗봇 대화(과거 대화 내용) 요약/재설명/참고가 필요한 경우

입력:
- primary_route: 규칙 기반 라우터가 예측한 1차 후보 (chit/db/disease/drug/web/history 중 하나)
- user_message: 사용자의 실제 질문

출력 포맷(반드시 이렇게만):
{
  "routes": ["agent_name1", "agent_name2", ...]
}

규칙:
- primary_route는 최대한 포함하려고 노력한다.
- 같은 에이전트를 여러 번 넣지 않는다.
- 특별히 복합 정보가 필요하지 않다면 1개만 선택해도 된다.
- 예시:
    - 개인 진료 기록 + 약 정보 => ["db", "drug"]
    - 약 정보가 메인인데 최신 이슈도 중요한 경우 => ["drug", "web"]
    - 과거 대화 내용이 중요해 보이는 경우 => ["history"]
    - 일반 잡담/개발 질문 => ["chit"]
"""


def _plan_routes_with_llm(user_message: str, primary_route: RouteName) -> List[RouteName]:
    if not user_message:
        return [primary_route]

    planner_user_message = (
        "아래 정보를 보고 어떤 에이전트들을 어떤 순서로 호출할지 결정해줘.\n\n"
        f"[primary_route]\n{primary_route}\n\n"
        f"[user_message]\n{user_message}\n\n"
        "반드시 JSON으로만 응답해야 한다. 예: {\"routes\": [\"db\", \"drug\"]}"
    )

    try:
        raw = call_llm(
            system_prompt=SUPERVISOR_PLANNER_SYSTEM_PROMPT,
            user_message=planner_user_message,
            context=None,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[SUPERVISOR] planner LLM 호출 실패: {e!r}")
        return [primary_route]

    if not raw:
        return [primary_route]

    try:
        data = json.loads(raw)
        routes_raw = data.get("routes", [])
        if not isinstance(routes_raw, list):
            raise ValueError("routes 필드가 list가 아님")

        allowed: List[RouteName] = ["chit", "db", "disease", "drug", "web", "history"]
        cleaned: List[RouteName] = []
        for r in routes_raw:
            if isinstance(r, str) and r in allowed and r not in cleaned:
                cleaned.append(r)

        if not cleaned:
            return [primary_route]

        return cleaned
    except Exception as e:
        print(f"[SUPERVISOR] planner JSON 파싱 실패: {e!r}  raw={raw!r}")
        return [primary_route]


# =========================================================
# 3) 라우트 → 에이전트 매핑
# =========================================================

def _run_agent(route: RouteName, state: ChatState) -> ChatState:
    if route == "chit":
        return chit_agent.run(state)
    if route == "db":
        return db_agent.run(state)
    if route == "disease":
        return disease_agent.run(state)
    if route == "drug":
        return drug_agent.run(state)
    if route == "web":
        return web_agent.run(state)
    if route == "history":
        return history_agent.run(state)

    return chit_agent.run(state)


# =========================================================
# 4) Final Generator (최종 답변 종합기)
# =========================================================

def _generate_final_answer(state: ChatState) -> ChatState:
    user_message = _get_last_user_message(state)
    context_list = state.get("context_list", [])

    if not context_list:
        answer = "죄송합니다. 관련 정보를 수집하지 못했습니다."
        state["answer"] = answer
        state["sources"] = []
        state["messages"].append({
            "role": "assistant",
            "content": answer,
            "meta": {"agent": "supervisor_final"},
        })
        return state

    combined_context_parts = []
    all_sources = []
    seen_source_urls = set()

    for item in context_list:
        agent_name = item.get("agent", "unknown")
        ctx = item.get("context", "")
        srcs = item.get("sources", [])
        
        if ctx:
            combined_context_parts.append(f"[{agent_name} 수집 정보]\n{ctx}")
        
        for src in srcs:
            url = src.get("url")
            if url:
                if url not in seen_source_urls:
                    seen_source_urls.add(url)
                    all_sources.append(src)
            else:
                all_sources.append(src)

    combined_context = "\n\n---\n\n".join(combined_context_parts)

    answer = call_llm(
        system_prompt=FINAL_GENERATOR_PROMPT,
        user_message=user_message,
        context=combined_context,
    )

    state["answer"] = answer
    state["sources"] = all_sources
    
    state["messages"].append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "agent": "supervisor_final",
            "agents_used": [item.get("agent") for item in context_list]
        },
    })
    
    return state


# =========================================================
# 5) 오케스트레이터 엔트리 포인트
# =========================================================

@traceable(name="orchestrator")
def run_orchestrator(state: ChatState) -> ChatState:
    """
    1) route_supervisor 로 1차 route 후보(primary) 정하기
    2) _plan_routes_with_llm 로 에이전트 실행 순서 결정
    3) 결정된 에이전트들을 순차 실행하여 context_list 에 정보 수집
    4) Final Generator를 통해 취합하여 최종 답변 생성 
       (단, chit_agent만 단독 실행된 경우 불필요한 LLM 호출을 줄이기 위해 바로 답변 반환)
    """
    user_message = _get_last_user_message(state)
    if not user_message:
        return chit_agent.run(state)

    primary_route = route_supervisor(state)

    planned_routes = _plan_routes_with_llm(
        user_message=user_message,
        primary_route=primary_route,
    )

    print(f"[SUPERVISOR] primary={primary_route}, planned_routes={planned_routes}")

    current_state = state
    # 정보 수집용 배열 초기화
    current_state["context_list"] = []

    for route in planned_routes:
        current_state = _run_agent(route, current_state)

    # 예외: chit_agent가 단독 호출된 경우 (잡담/일반대화)
    if len(planned_routes) == 1 and planned_routes[0] == "chit":
        # Final Generator 호출 없이 chit_agent의 응답을 그대로 사용
        context_list = current_state.get("context_list", [])
        if context_list:
            last_chit_context = context_list[-1]["context"]
            current_state["answer"] = last_chit_context
            current_state["sources"] = []
            current_state["messages"].append(
                {
                    "role": "assistant",
                    "content": last_chit_context,
                    "meta": {"agent": "chit_agent"},
                }
            )
        return current_state

    # 그 외 정보 수집을 수행한 경우: Final Generator 실행
    current_state = _generate_final_answer(current_state)

    return current_state
