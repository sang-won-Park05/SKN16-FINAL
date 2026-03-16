# ============================================================
# STT ROUTER (테스트용: user_id = 1 고정)
# ============================================================

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import httpx
import os

from database import get_db
from crud.stt_crud import (
    create_stt_job,
    get_stt_job,
    update_stt_result,
)
from schemas.stt_schemas import (
    STTAnalyzeResponse,
    STTStatusResponse,
    STTResultInput,
)
from utils.redis_store import get_stt_status, set_stt_status

router = APIRouter(prefix="/stt", tags=["STT"])

# 테스트용 FAKE USER
FAKE_USER_ID = 1

# STT 서버 URL (Docker: stt, 로컬: localhost:8002)
STT_SERVER_URL = os.getenv("STT_SERVER_URL", "http://localhost:8002")


# ------------------------------------------------------------
# 1) STT 분석 시작 (stt_id 발급 + pending)
# ------------------------------------------------------------
@router.post("/analyze", response_model=STTAnalyzeResponse)
async def analyze_stt(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    음성 파일을 받아 STT 작업(stt_id)을 생성하고
    STT 서버로 전달하여 백그라운드 처리.
    """

    user_id = FAKE_USER_ID

    # STTJob 생성
    stt_item = create_stt_job(db, user_id=user_id)
    set_stt_status(
        stt_item.stt_id,
        {
            "stt_id": stt_item.stt_id,
            "user_id": user_id,
            "status": stt_item.status,
        },
    )

    # STT 서버로 파일 전송
    try:
        file_content = await file.read()
        await file.seek(0)  # 파일 포인터 리셋

        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            data = {"stt_id": stt_item.stt_id}

            response = await client.post(
                f"{STT_SERVER_URL}/stt/process",
                files=files,
                data=data
            )

            if response.status_code != 200:
                print(f"⚠️ STT 서버 요청 실패: {response.status_code}")

    except Exception as e:
        print(f"❌ STT 서버 연결 실패: {e}")
        # 에러 발생해도 stt_id는 반환 (status=pending 유지)

    return STTAnalyzeResponse(
        user_id=user_id,
        stt_id=stt_item.stt_id,
        status=stt_item.status,
    )


# ------------------------------------------------------------
# 2) STT 상태 조회 (프론트 폴링)
# ------------------------------------------------------------
@router.get("/{stt_id}/status", response_model=STTStatusResponse)
def get_status(
    stt_id: str,
    db: Session = Depends(get_db),
):
    """
    STTJob의 현재 상태 및 결과를 조회.
    프론트에서 주기적으로 호출해서
    status가 done이 되면 diagnosis/symptoms/notes/date를 사용.
    """

    job = get_stt_job(db, stt_id)
    cached = get_stt_status(stt_id)

    if cached:
        if job and cached.get("status") in {"done", "error"} and job.status != cached.get("status"):
            updated = update_stt_result(
                db,
                stt_id,
                {
                    "status": cached.get("status"),
                    "diagnosis": cached.get("diagnosis"),
                    "symptoms": cached.get("symptoms"),
                    "notes": cached.get("notes"),
                    "date": cached.get("date"),
                },
            )
            if updated:
                job = updated

        payload = {
            "stt_id": stt_id,
            "status": cached.get("status") or (job.status if job else "pending"),
            "user_id": cached.get("user_id") or (job.user_id if job else None),
            "diagnosis": cached.get("diagnosis") or (job.diagnosis if job else None),
            "symptoms": cached.get("symptoms") or (job.symptoms if job else None),
            "notes": cached.get("notes") or (job.notes if job else None),
            "date": cached.get("date") or (job.date if job else None),
        }
        return STTStatusResponse(**payload)

    if not job:
        raise HTTPException(status_code=404, detail="STT job not found")

    return job


# ------------------------------------------------------------
# 3) Whisper 결과 저장 (STT 담당자 → 백엔드)
# ------------------------------------------------------------
@router.post("/{stt_id}/result")
async def receive_stt_result(
    stt_id: str,
    result: STTResultInput,
    db: Session = Depends(get_db),
):
    """
    STT 담당자가 Whisper 실행 후,
    stt_id 기준으로 결과를 백엔드에 넘기는 엔드포인트.
    """

    updated = update_stt_result(db, stt_id, result.dict())
    if not updated:
        raise HTTPException(status_code=404, detail="STT job not found")

    set_stt_status(
        stt_id,
        {
            "stt_id": stt_id,
            "user_id": updated.user_id,
            "status": updated.status,
            "diagnosis": updated.diagnosis,
            "symptoms": updated.symptoms,
            "notes": updated.notes,
            "date": updated.date,
        },
    )

    return {"message": "result saved", "stt_id": stt_id}
