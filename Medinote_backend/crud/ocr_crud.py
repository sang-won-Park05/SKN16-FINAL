# crud/ocr_crud.py

import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import UploadFile
from sqlalchemy.orm import Session

from models import File, OCRJob
from schemas.ocr_schemas import VisitFormSchema
from utils.redis_store import (
    get_ocr_parse,
    get_ocr_result,
    set_ocr_job_status,
    set_ocr_parse,
    set_ocr_result,
)

UPLOAD_DIR = "uploads/ocr"


# -----------------------------
# 파일 저장
# -----------------------------
def save_file(upload: UploadFile) -> tuple[str, str]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(upload.filename or "")[1]
    new_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, new_name)

    upload.file.seek(0)
    content = upload.file.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    file_hash = hashlib.sha256(content).hexdigest()

    with open(file_path, "wb") as f:
        f.write(content)

    upload.file.seek(0)

    return file_path, file_hash


# -----------------------------
# File 레코드 생성
# -----------------------------
def create_file_record(
    db: Session,
    user_id: int,
    upload: UploadFile,
    path: str
) -> File:
    size = os.path.getsize(path)

    file = File(
        user_id=user_id,
        path=path,
        original_name=upload.filename,
        mime_type=upload.content_type,
        size=size
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


# -----------------------------
# Dummy OCR Model
# (모델 작업자가 이 함수만 교체하면 됨)
# -----------------------------
def run_ocr_model(path: str) -> str:
    return f"OCR 더미 결과입니다. (path={path})"


# -----------------------------
# OCR 실행 + DB 저장
# -----------------------------
def run_ocr_and_save(
    db: Session,
    user_id: int,
    upload_file: UploadFile,
    source_type: str,
    visit_id: Optional[int] = None,   # visit_id는 Nullable
):
    # 1) 파일 저장
    path, file_hash = save_file(upload_file)

    # 2) File 레코드 생성
    file_obj = create_file_record(db, user_id, upload_file, path)

    # 3) OCRJob 레코드 생성 (RUNNING)
    ocr = OCRJob(
        user_id=user_id,
        file_id=file_obj.file_id,
        visit_id=visit_id,         # 지금은 Visit에만 FK 있음
        source_type=source_type,
        status="RUNNING",
        created_at=datetime.utcnow()
    )
    db.add(ocr)
    db.commit()
    db.refresh(ocr)
    set_ocr_job_status(
        ocr.ocr_id,
        {
            "ocr_id": ocr.ocr_id,
            "file_id": ocr.file_id,
            "user_id": ocr.user_id,
            "source_type": ocr.source_type,
            "status": ocr.status,
            "text": ocr.text,
            "visit_id": ocr.visit_id,
            "created_at": ocr.created_at,
            "completed_at": ocr.completed_at,
        },
    )

    # 4) OCR 모델 실행
    try:
        cached = get_ocr_result(source_type, file_hash)
        if cached and cached.get("text") is not None:
            text = cached.get("text", "")
        else:
            text = run_ocr_model(path)
            set_ocr_result(
                source_type,
                file_hash,
                {
                    "status": "DONE",
                    "text": text,
                },
            )
        ocr.text = text
        ocr.status = "DONE"
    except Exception as e:
        ocr.status = "FAILED"
        ocr.text = f"OCR ERROR: {e}"

    ocr.completed_at = datetime.utcnow()

    db.add(ocr)
    db.commit()
    db.refresh(ocr)
    set_ocr_job_status(
        ocr.ocr_id,
        {
            "ocr_id": ocr.ocr_id,
            "file_id": ocr.file_id,
            "user_id": ocr.user_id,
            "source_type": ocr.source_type,
            "status": ocr.status,
            "text": ocr.text,
            "visit_id": ocr.visit_id,
            "created_at": ocr.created_at,
            "completed_at": ocr.completed_at,
        },
    )

    return ocr


# -----------------------------
# OCR Raw → Visit/Prescription 폼 구조화 (LLM 자리)
# -----------------------------
def parse_ocr_text_to_visit(text: str) -> VisitFormSchema:
    """
    OCR result text → Visit/Prescription 폼 자동 구조화

    지금은 dummy.
    LLM 연동 시 여기를 교체하면 됨.
    """

    cached = get_ocr_parse("visit", text)
    if cached:
        return VisitFormSchema(**cached)

    dummy = {
        "hospital": "",
        "doctor_name": "",
        "symptom": text[:30],  # 일단 raw text 앞부분만 사용
        "opinion": "",
        "diagnosis_code": "",
        "diagnosis_name": "",
        "date": str(datetime.today().date())
    }

    parsed = VisitFormSchema(**dummy)
    set_ocr_parse("visit", text, parsed.dict())
    return parsed
