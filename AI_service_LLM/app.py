# C:\Users\playdata\Desktop\SKN16-FINAL-1Team\AI_service_LLM\app.py

from __future__ import annotations

import os
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 🔹 DB 저장/조회용 레포지토리
from chatbot.core.chat_repository import (
    upsert_session_with_log,
    list_sessions as db_list_sessions,
    get_session_messages as db_get_session_messages,
    delete_session as db_delete_session,
    delete_all_sessions as db_delete_all_sessions,
)

# 🔹 ChatState & Supervisor(오케스트레이터)
from chatbot.core.state import ChatState
from chatbot.core.supervisor import run_orchestrator

# 🔹 건강 분석용 (db_agent 로직 재사용)
from chatbot.core.user_repository import (
    get_user_profile,
    get_allergies,
    get_chronic_diseases,
    get_acute_diseases,
    get_drugs,
    get_prescriptions,
    get_visits,
)
from chatbot.core.llm import call_llm
from chatbot.core.prompts import HEALTH_ANALYSIS_PROMPT
from chatbot.core.redis_store import (
    get_cached_analysis,
    get_cached_query_response,
    set_cached_analysis,
    set_cached_query_response,
)

load_dotenv()

# ============================================
# 공통 설정
# ============================================

# ⚠️ 인증 연동 전이므로 기본 사용자 ID를 환경변수로 받고, 없으면 1을 사용
DEFAULT_USER_ID = int(
    os.getenv("LLM_DEFAULT_USER_ID")
    or os.getenv("DEFAULT_USER_ID")
    or "1"
)


def _default_user_id(user_id: int | None = None) -> int:
    """
    - 인자가 있으면 그대로 사용
    - 없으면 DEFAULT_USER_ID 사용
    """
    return user_id if user_id is not None else DEFAULT_USER_ID


MEDINOTE_SYSTEM_PROMPT = """
너는 '메디노트' 서비스의 AI 건강 챗봇이다.
- 사용자의 증상, 복용 중인 약, 병원 진료 기록 등을 바탕으로 일반적인 건강 정보와 생활 수칙을 안내한다.
- 의사/약사가 아니며, '진단', '처방', '특정 약 복용 지시'는 절대 내리지 않는다.
- 위험 신호(심한 통증, 호흡곤란, 의식 변화 등)가 의심되면 즉시 병원·응급실 방문을 권한다.
- 사용자가 이해하기 쉽게 한국어로, 친절하고 차분하게 설명한다.
"""

# ============================================
# Pydantic 모델들 (프론트/백엔드와 1:1 매핑)
# ============================================


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
    - session_id: 0 이면 새 세션, 그 외에는 기존 세션 이어쓰기
    - query: 사용자 질문
    """
    session_id: int       # 0이면 새 세션
    query: str            # 사용자 질문 텍스트
    # 🔥 user_id 는 이제 바디에서 받지 않고 서버 내부에서 기본값 사용


class ChatQueryResponse(BaseModel):
    session_id: int
    answer: str
    # 🔥 이번 턴에서 사용된 출처 리스트
    sources: List[ChatSource] = []


class HealthAnalysisResponse(BaseModel):
    """건강 분석 리포트 응답 (챗봇 이력 저장 X)"""
    analysis: str


class SessionItem(BaseModel):
    session_id: int
    title: str
    created_at: datetime   # Swagger example: "2025-12-05T09:35:20.871Z"


class SessionsResponse(BaseModel):
    sessions: List[SessionItem]


class SessionMessage(BaseModel):
    role: str              # "user" | "assistant"
    content: str
    created_at: datetime   # Swagger example: ISO8601 datetime
    # 🔥 메인 백엔드와 맞추기 위해 optional sources 추가
    sources: Optional[List[ChatSource]] = None


class SessionDetailResponse(BaseModel):
    session_id: int
    messages: List[SessionMessage]


# ============================================
# FastAPI 앱 & CORS 설정
# ============================================

app = FastAPI(title="MediNote AI LLM Service", version="0.2.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]

# 환경변수에서 FRONTEND_URL 추가
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 개발 중엔 ["*"] 도 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# (옵션) DB 기반 컨텍스트 빌더
# ============================================

def _build_context_from_db(
    session_id: int,
    user_id: int | None = None,
) -> str | None:
    """
    DB에서 해당 세션의 대화 내역을 가져와서
    LLM에 넘길 context 문자열로 변환.
    (최근 10개 메시지만 사용)
    """
    if not session_id or session_id == 0:
        return None

    rows = db_get_session_messages(
        session_id=session_id,
        user_id=_default_user_id(user_id),
    )
    if not rows:
        return None

    last_msgs = rows[-10:]

    lines: List[str] = []
    for m in last_msgs:
        prefix = "사용자" if m["role"] == "user" else "챗봇"
        lines.append(f"[{prefix}] {m['content']}")

    return "\n".join(lines)


# ============================================
# 기본 health 체크
# ============================================

@app.get("/health", tags=["default"])
async def health_check():
    return {"status": "ok"}


# ============================================
# POST /chatbot/query  (⭢ LangGraph + DB 저장)
# ============================================

@app.post("/chatbot/query", response_model=ChatQueryResponse, tags=["chatbot"])
async def post_chatbot_query(payload: ChatQueryRequest):
    """
    - payload.session_id == 0 또는 세션 없음 → 새 세션 생성 + 첫 로그 저장
    - payload.session_id != 0              → 해당 세션에 로그 append

    흐름:
      1) ChatState 구성 (user_id / session_id / messages)
      2) run_orchestrator(state) 실행 → 여러 에이전트 조합
      3) result 에서 answer / sources 추출
      4) upsert_session_with_log(...) 로 세션/로그 저장
      5) session_id + answer + sources 반환
    """
    # 🔥 이제 body 에서 user_id 안 받고, 서버 내부 기본값 사용
    user_id = _default_user_id()

    # 1) ChatState 구성
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

    # 기존 세션이면 state에 힌트로 넣어준다.
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
        sources: List[ChatSource] = [
            ChatSource(**s) for s in sources_raw
            if isinstance(s, dict)
        ]
    else:
        # 2) 오케스트레이터 실행
        try:
            new_state = run_orchestrator(state)
        except Exception as e:
            print(f"[LLM ERROR] session_id={payload.session_id} error={e!r}")
            raise HTTPException(
                status_code=500,
                detail=(
                    "현재 챗봇 엔진에 문제가 발생하여 답변을 생성할 수 없습니다. "
                    "잠시 후 다시 시도해 주세요."
                ),
            )

        # 3) answer / sources 추출
        answer_text = new_state.get("answer") or ""

        # safety: answer 가 비어 있으면 마지막 assistant 메시지에서 fallback
        if not answer_text:
            msgs = new_state.get("messages") or []
            if msgs and msgs[-1].get("role") == "assistant":
                answer_text = msgs[-1].get("content", "")

        if not answer_text:
            answer_text = (
                "죄송합니다. 현재는 적절한 답변을 생성하지 못했습니다. "
                "질문을 조금 더 구체적으로 말씀해 주시면 도움이 됩니다."
            )

        sources_raw = new_state.get("sources") or []
        sources = [
            ChatSource(**s) for s in sources_raw
            if isinstance(s, dict)
        ] if sources_raw else []
        set_cached_query_response(
            user_id=user_id,
            session_id=payload.session_id,
            query=payload.query,
            answer=answer_text,
            sources=[s.model_dump() for s in sources] if sources else None,
        )

    # DB에 저장할 수 있도록 순수 dict 리스트로 변환
    sources_for_db = [s.model_dump() for s in sources] if sources else None

    # 4) DB 에 세션 + 로그 저장
    used_session_id = upsert_session_with_log(
        session_id=payload.session_id,
        user_id=user_id,
        query=payload.query,
        answer=answer_text,
        sources=sources_for_db,
    )

    # 5) 프론트/백엔드로 session_id + answer + sources 반환
    return ChatQueryResponse(
        session_id=used_session_id,
        answer=answer_text,
        sources=sources,
    )


# ============================================
# GET /chatbot/sessions  (세션 목록)
# ============================================

@app.get("/chatbot/sessions", response_model=SessionsResponse, tags=["chatbot"])
async def get_chatbot_sessions():
    """
    세션 목록 조회.
    DB 에러가 나도 서비스 전체가 죽지 않도록 try/except 로 감싸고,
    실패 시에는 빈 배열을 반환한다.
    """
    try:
        target_user_id = _default_user_id()
        rows = db_list_sessions(user_id=target_user_id, limit=50, order="desc")
        # rows 예: [{"session_id":1,"title":"...","created_at":"2025-12-05T09:35:20.871Z"}, ...]
        sessions = [SessionItem(**row) for row in rows]
        return SessionsResponse(sessions=sessions)
    except Exception as e:
        print(f"[SESSIONS ERROR] user_id={_default_user_id()} error={e!r}")
        return SessionsResponse(sessions=[])


# ============================================
# DELETE /chatbot/sessions  (해당 유저 전체 삭제)
# ============================================

@app.delete("/chatbot/sessions", response_model=str, tags=["chatbot"])
async def delete_all_chatbot_sessions():
    db_delete_all_sessions(user_id=_default_user_id())
    return "모든 챗봇 세션을 삭제했습니다."


# ============================================
# GET /chatbot/sessions/{session_id}  (세션 상세)
# ============================================

@app.get(
    "/chatbot/sessions/{session_id}",
    response_model=SessionDetailResponse,
    tags=["chatbot"],
)
async def get_chatbot_session_detail(session_id: int):
    """
    특정 세션의 전체 메시지 조회.
    에러 시 404/500 대신 깔끔한 메시지로 정리.
    """
    try:
        target_user_id = _default_user_id()
        rows = db_get_session_messages(
            session_id=session_id,
            user_id=target_user_id,
        )
    except Exception as e:
        print(f"[SESSION DETAIL ERROR] session_id={session_id} error={e!r}")
        raise HTTPException(status_code=500, detail="세션 정보를 불러오지 못했습니다.")

    if not rows:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # rows 안에 created_at 은 ISO 문자열 / datetime 둘 다 허용
    # sources 컬럼이 없으면 Pydantic 이 기본값(None) 채움
    messages = [SessionMessage(**row) for row in rows]

    return SessionDetailResponse(
        session_id=session_id,
        messages=messages,
    )


# ============================================
# DELETE /chatbot/sessions/{session_id}  (단일 세션 삭제)
# ============================================

@app.delete(
    "/chatbot/sessions/{session_id}",
    response_model=str,
    tags=["chatbot"],
)
async def delete_one_chatbot_session(session_id: int):
    deleted = db_delete_session(
        session_id=session_id,
        user_id=_default_user_id(),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    return f"{session_id}번 세션을 삭제했습니다."


# ============================================
# POST /chatbot/analysis  (건강 분석 리포트 - 챗봇 이력 저장 X)
# ============================================

@app.post("/chatbot/analysis", response_model=HealthAnalysisResponse, tags=["chatbot"])
async def post_health_analysis():
    """
    건강 분석 리포트 생성 (챗봇 대화 이력 저장 X)
    - db_agent 로직 재사용하여 사용자 건강 데이터 조회
    - LLM으로 분석 리포트 생성
    - chat_log에 저장하지 않음
    """
    user_id = _default_user_id()

    # 1) 백엔드에서 건강 데이터 조회 (병렬 처리로 속도 개선)
    import concurrent.futures

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            future_profile = executor.submit(get_user_profile, user_id)
            future_allergies = executor.submit(get_allergies, user_id)
            future_chronic = executor.submit(get_chronic_diseases, user_id)
            future_acute = executor.submit(get_acute_diseases, user_id)
            future_drugs = executor.submit(get_drugs, user_id)
            future_prescriptions = executor.submit(get_prescriptions, user_id)
            future_visits = executor.submit(get_visits, user_id)

            profile = future_profile.result()
            allergies = future_allergies.result()
            chronic = future_chronic.result()
            acute = future_acute.result()
            drugs = future_drugs.result()
            prescriptions = future_prescriptions.result()
            visits = future_visits.result()   
    except Exception as e:
        print(f"[ANALYSIS ERROR] user_id={user_id} error={e!r}")
        raise HTTPException(
            status_code=500,
            detail="건강 정보를 불러오는 중 오류가 발생했습니다."
        )

    # 2) LLM에 전달할 컨텍스트 구성
    context_blocks = []

    if profile:
        # BMI 계산
        height = profile.get('height')
        weight = profile.get('weight')
        bmi_str = ""
        if height and weight:
            h_m = float(height) / 100
            bmi = float(weight) / (h_m * h_m)
            bmi_str = f" (BMI: {bmi:.1f})"

        context_blocks.append(
            f"[건강 프로필]\n"
            f"출생: {profile.get('birth')}\n"
            f"성별: {profile.get('gender')}\n"
            f"혈액형: {profile.get('blood_type')}\n"
            f"키: {profile.get('height')} cm\n"
            f"몸무게: {profile.get('weight')} kg{bmi_str}\n"
            f"음주: {profile.get('drinking')}\n"
            f"흡연: {profile.get('smoking')}"
        )

    if allergies:
        context_blocks.append(
            "[알레르기]\n" + "\n".join([a.get('allergy_name', '') for a in allergies])
        )

    if chronic:
        context_blocks.append(
            "[만성 질환]\n" + "\n".join([f"{c.get('disease_name')} ({c.get('note', '')})" for c in chronic])
        )

    if acute:
        context_blocks.append(
            "[급성 질환]\n" + "\n".join([f"{a.get('disease_name')} ({a.get('note', '')})" for a in acute])
        )

    if drugs:
        drug_lines = [f"{d.get('med_name')} {d.get('dose')}{d.get('unit')} ({d.get('schedule')})" for d in drugs]
        context_blocks.append("[복용 중인 약]\n" + "\n".join(drug_lines))

    if prescriptions:
        pres_lines = [f"{p.get('med_name')} ({p.get('start_date')}~{p.get('end_date')})" for p in prescriptions]
        context_blocks.append("[처방 이력]\n" + "\n".join(pres_lines))

    if visits:
        visit_lines = [f"{v.get('hospital')} {v.get('dept')} - {v.get('diagnosis_name')} ({v.get('date')})" for v in visits]
        context_blocks.append("[진료 기록]\n" + "\n".join(visit_lines))

    medical_context = "\n\n".join(context_blocks) if context_blocks else "등록된 건강 정보가 없습니다."

    # 3) LLM 호출
    cached_analysis = get_cached_analysis(user_id=user_id, context=medical_context)
    if cached_analysis:
        return HealthAnalysisResponse(analysis=cached_analysis)

    try:
        analysis = call_llm(
            system_prompt=HEALTH_ANALYSIS_PROMPT,
            user_message="내 건강 상태를 분석해주세요.",
            context=medical_context,
        )
    except Exception as e:
        print(f"[ANALYSIS LLM ERROR] user_id={user_id} error={e!r}")
        raise HTTPException(
            status_code=500,
            detail="건강 분석 중 오류가 발생했습니다."
        )

    if not analysis:
        analysis = "건강 정보가 부족하여 분석을 수행할 수 없습니다."

    set_cached_analysis(user_id=user_id, context=medical_context, analysis=analysis)

    # 4) 응답 반환 (chat_log 저장 X)
    return HealthAnalysisResponse(analysis=analysis)
