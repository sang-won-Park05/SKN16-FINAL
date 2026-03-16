"""
STT Processing Router
백그라운드로 STT 처리 + 요약 생성 + 백엔드로 결과 전송
"""
import os
import time
import tempfile
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from datetime import datetime
import httpx

from core.engine.openai_engine import OpenAIWhisperSTT
from core.summarize import generate_summary
from core.redis_queue import (
    dequeue_stt_job,
    enqueue_stt_job,
    is_redis_available,
    set_stt_status,
)

router = APIRouter(prefix="/stt", tags=["STT"])

# STT 엔진 (OpenAI Whisper API)
stt_engine = OpenAIWhisperSTT(model="whisper-1")

# 백엔드 URL (환경 변수로 관리 가능)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_worker_thread: threading.Thread | None = None
_worker_stop_event = threading.Event()
_worker_lock = threading.Lock()


def _build_status_payload(stt_id: str, status: str, **extra) -> dict:
    payload = {
        "stt_id": stt_id,
        "status": status,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def process_stt_task(stt_id: str, file_path: str):
    """
    백그라운드에서 실행되는 STT 처리 함수
    처리 완료 후 백엔드로 결과 POST

    Args:
        stt_id: STT 작업 ID (백엔드에서 생성)
        file_path: 임시 저장된 오디오 파일 경로
    """
    try:
        total_start = time.time()
        print(f"🎙️ [{stt_id}] STT 처리 시작...")
        set_stt_status(stt_id, _build_status_payload(stt_id, "processing"))

        # 1. STT 처리
        stt_start = time.time()
        result = stt_engine.transcribe(file_path)
        transcript_text = result.get("text", "")
        stt_time = time.time() - stt_start

        print(f"✅ [{stt_id}] STT 완료 ({stt_time:.1f}초): {len(transcript_text)} 글자")

        # 2. 요약 생성
        if transcript_text.strip():
            print(f"🤖 [{stt_id}] 요약 생성 중...")
            summary_start = time.time()
            summary_result = generate_summary(transcript_text)
            summary_time = time.time() - summary_start
            print(f"✅ [{stt_id}] 요약 완료 ({summary_time:.1f}초)")
            result_payload = _build_status_payload(
                stt_id,
                "done",
                symptoms=summary_result["symptoms"],
                diagnosis=summary_result["diagnosis"],
                notes=summary_result["notes"],
                date=datetime.now().strftime("%Y-%m-%d"),
            )
            set_stt_status(stt_id, result_payload)

            # 3. 백엔드로 결과 POST
            print(f"📤 [{stt_id}] 백엔드로 결과 전송 중...")

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{BACKEND_URL}/stt/{stt_id}/result",
                    json=result_payload
                )

                if response.status_code == 200:
                    total_time = time.time() - total_start
                    print(f"✅ [{stt_id}] 백엔드 업데이트 성공")
                    print(f"⏱️ [{stt_id}] 전체 처리 완료: {total_time:.1f}초")
                else:
                    print(f"⚠️ [{stt_id}] 백엔드 업데이트 실패: {response.status_code}")

        else:
            print(f"⚠️ [{stt_id}] 텍스트가 비어있어 요약 생략")
            error_payload = _build_status_payload(
                stt_id,
                "error",
                symptoms="",
                diagnosis="",
                notes="음성 인식 실패 (빈 텍스트)",
                date=datetime.now().strftime("%Y-%m-%d"),
            )
            set_stt_status(stt_id, error_payload)

            # 빈 텍스트도 백엔드에 알림
            with httpx.Client(timeout=30.0) as client:
                client.post(
                    f"{BACKEND_URL}/stt/{stt_id}/result",
                    json=error_payload
                )

    except Exception as e:
        print(f"❌ [{stt_id}] 처리 실패: {e}")
        error_payload = _build_status_payload(
            stt_id,
            "error",
            symptoms="",
            diagnosis="",
            notes=f"STT 처리 중 에러: {str(e)}",
            date=datetime.now().strftime("%Y-%m-%d"),
        )
        set_stt_status(stt_id, error_payload)

        # 에러 발생 시 백엔드에 알림
        try:
            with httpx.Client(timeout=30.0) as client:
                client.post(
                    f"{BACKEND_URL}/stt/{stt_id}/result",
                    json=error_payload
                )
        except:
            pass

    finally:
        # 4. 임시 파일 삭제
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ [{stt_id}] 임시 파일 삭제 완료")
        except Exception as e:
            print(f"⚠️ [{stt_id}] 임시 파일 삭제 실패: {e}")


def _worker_loop():
    print("✅ STT Redis worker started")
    while not _worker_stop_event.is_set():
        job = dequeue_stt_job(timeout=1)
        if not job:
            continue

        stt_id = job.get("stt_id")
        file_path = job.get("file_path")
        if not stt_id or not file_path:
            continue

        process_stt_task(stt_id, file_path)


def start_stt_worker():
    global _worker_thread

    if not is_redis_available():
        print("⚠️ Redis unavailable - STT queue worker disabled")
        return

    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return

        _worker_stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="stt-redis-worker",
            daemon=True,
        )
        _worker_thread.start()


def stop_stt_worker():
    _worker_stop_event.set()


@router.post("/process")
async def process_stt(
    background_tasks: BackgroundTasks,
    stt_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    백엔드에서 호출하는 STT 처리 엔드포인트

    백엔드가 이미 stt_id를 생성하고 DB에 pending 상태로 저장한 후,
    이 엔드포인트로 stt_id + 파일을 전송합니다.

    Args:
        stt_id: 백엔드에서 생성한 STT 작업 ID
        file: 업로드된 오디오 파일
        background_tasks: FastAPI BackgroundTasks

    Returns:
        {"message": "Processing started", "stt_id": stt_id}
    """
    try:
        # 1. 임시 파일로 저장
        suffix = Path(file.filename).suffix if file.filename else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        print(f"📁 [{stt_id}] 파일 저장: {temp_file_path} ({len(content)} bytes)")

        # 2. Redis 큐 또는 FastAPI BackgroundTasks 로 처리 시작
        set_stt_status(stt_id, _build_status_payload(stt_id, "queued"))
        if not enqueue_stt_job(stt_id, temp_file_path):
            background_tasks.add_task(process_stt_task, stt_id, temp_file_path)

        return {
            "message": "STT processing queued",
            "stt_id": stt_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/status")
async def check_status():
    """STT 서버 상태 확인"""
    return {
        "status": "running",
        "service": "STT Processing API",
        "port": 8002
    }
