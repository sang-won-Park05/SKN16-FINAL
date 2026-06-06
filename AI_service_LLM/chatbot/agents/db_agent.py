# AI_service_LLM/chatbot/agents/db_agent.py

from __future__ import annotations

from typing import List, Dict

from ..core.state import ChatState
from ..core.tracing import traceable
from ..core.user_repository import (
    get_user_profile,
    get_allergies,
    get_chronic_diseases,
    get_acute_diseases,
    get_drugs,
    get_prescriptions,
    get_visits,
)


@traceable(name="db_agent")
def run(state: ChatState) -> ChatState:

    user_id = state.get("user_id")

    if "context_list" not in state:
        state["context_list"] = []

    # ------------------------------------------------
    # 0) user_id 없을 때
    # ------------------------------------------------
    if not user_id:
        state["context_list"].append(
            {
                "agent": "db_agent",
                "context": "현재 사용자 정보를 확인할 수 없어 의료 기록을 불러올 수 없습니다. 로그인이 되어 있는지 확인해주세요.",
                "sources": [],
                "meta": {"error": "no_user_id"},
            }
        )
        return state

    # ------------------------------------------------
    # 1) 백엔드 데이터 가져오기
    # ------------------------------------------------
    try:
        profile = get_user_profile(user_id)
        allergies = get_allergies(user_id)
        chronic = get_chronic_diseases(user_id)
        acute = get_acute_diseases(user_id)
        drugs = get_drugs(user_id)
        prescriptions = get_prescriptions(user_id)
        visits = get_visits(user_id)

    except Exception as e:
        state["context_list"].append(
            {
                "agent": "db_agent",
                "context": f"의료 기록 조회 중 오류가 발생했습니다: {e}",
                "sources": [],
                "meta": {"error": str(e)},
            }
        )
        return state

    # ------------------------------------------------
    # 2) GPT에게 전달할 Context 구성
    # ------------------------------------------------

    context_blocks = []

    if profile:
        context_blocks.append(
            "## 건강 프로필\n"
            f"- 출생: {profile.get('birth')}\n"
            f"- 성별: {profile.get('gender')}\n"
            f"- 혈액형: {profile.get('blood_type')}\n"
            f"- 키: {profile.get('height')} cm\n"
            f"- 몸무게: {profile.get('weight')} kg\n"
            f"- 음주: {profile.get('drinking')}\n"
            f"- 흡연: {profile.get('smoking')}"
        )

    if allergies:
        context_blocks.append(
            "## 알레르기 목록\n"
            + "\n".join([f"- {a.get('allergy_name')}" for a in allergies])
        )

    if chronic:
        context_blocks.append(
            "## 만성 질환 목록\n"
            + "\n".join([f"- {c.get('disease_name')} (메모: {c.get('note')})" for c in chronic])
        )

    if acute:
        context_blocks.append(
            "## 급성 질환 목록\n"
            + "\n".join([f"- {a.get('disease_name')} (메모: {a.get('note')})" for a in acute])
        )

    if drugs:
        drug_lines = []
        for d in drugs:
            drug_lines.append(
                f"- {d.get('med_name')} | {d.get('dosage_form')} | "
                f"{d.get('dose')}{d.get('unit')} | 일정: {d.get('schedule')}"
            )
        context_blocks.append("## 복용 중인 약 목록\n" + "\n".join(drug_lines))

    if prescriptions:
        pres_lines = []
        for p in prescriptions:
            pres_lines.append(
                f"- {p.get('med_name')} | {p.get('dosage_form')} | "
                f"{p.get('dose')}{p.get('unit')} | 일정: {p.get('schedule')} "
                f"({p.get('start_date')}~{p.get('end_date')})"
            )
        context_blocks.append("## 처방 이력\n" + "\n".join(pres_lines))

    if visits:
        visit_lines = []
        for v in visits:
            visit_lines.append(
                f"- {v.get('hospital')} | {v.get('dept')} | "
                f"{v.get('diagnosis_name')} | {v.get('date')}"
            )
        context_blocks.append("## 진료 기록\n" + "\n".join(visit_lines))

    medical_context = "\n\n".join(context_blocks) if context_blocks else "사용자의 의료 기록이 없습니다."

    # ------------------------------------------------
    # 3) 결과 state에 push
    # ------------------------------------------------
    state["context_list"].append(
        {
            "agent": "db_agent",
            "context": medical_context,
            "sources": [],
            "meta": {
                "profile": bool(profile),
                "allergy_count": len(allergies or []),
                "drug_count": len(drugs or []),
                "prescription_count": len(prescriptions or []),
            },
        }
    )

    return state
