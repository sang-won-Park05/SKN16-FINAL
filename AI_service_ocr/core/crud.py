# core/crud.py
from __future__ import annotations

import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional, List

from fastapi import UploadFile
from sqlalchemy.orm import Session

from core.models import File, OCRJob
from core.schemas import VisitFormSchema, PrescriptionFormSchema
from core.paddle_pipeline import extract_text_from_image
from core.gpt_client import (
    parse_visit_form_from_ocr,
    parse_prescription_form_from_ocr,
)
from core.config import OCR_UPLOAD_DIR
from core.redis_store import (
    get_ocr_result,
    get_parse_result,
    set_ocr_job_status,
    set_ocr_result,
    set_parse_result,
)


# -----------------------------
# 파일 저장
# -----------------------------
def save_file(upload: UploadFile) -> tuple[str, str]:
    os.makedirs(OCR_UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(upload.filename or "")[1]
    new_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(OCR_UPLOAD_DIR, new_name)

    upload.file.seek(0)
    content = upload.file.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    file_hash = hashlib.sha256(content).hexdigest()
    with open(file_path, "wb") as f:
        f.write(content)

    upload.file.seek(0)

    print(f"[OCR] Saved upload to: {file_path}")
    return file_path, file_hash


# -----------------------------
# File 테이블 Row 생성
# -----------------------------
def create_file_record(
    db: Session,
    user_id: int,
    upload: UploadFile,
    path: str,
) -> File:
    size = os.path.getsize(path)

    file = File(
        user_id=user_id,
        path=path,
        original_name=upload.filename,
        mime_type=upload.content_type,
        size=size,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    print(f"[OCR] File row created: file_id={file.file_id}, size={size}")
    return file


# -----------------------------
# Paddle OCR 실행
# -----------------------------
def run_ocr_model(path: str) -> str:
    print(f"[OCR] Running Paddle OCR on: {path}")
    text = extract_text_from_image(path)
    print(f"[OCR] OCR text length: {len(text)}")
    print("[OCR] OCR text head:")
    print(text[:500])
    return text


# -----------------------------
# OCR 실행 + DB 저장
# -----------------------------
def run_ocr_and_save(
    db: Session,
    user_id: int,
    upload_file: UploadFile,
    source_type: str,
    visit_id: Optional[int] = None,
) -> OCRJob:
    path, file_hash = save_file(upload_file)

    file_obj = create_file_record(db, user_id, upload_file, path)

    ocr = OCRJob(
        user_id=user_id,
        file_id=file_obj.file_id,
        visit_id=visit_id,
        source_type=source_type,
        status="RUNNING",
        created_at=datetime.utcnow(),
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
    print(
        f"[OCR] OCRJob created: ocr_id={ocr.ocr_id}, "
        f"source_type={source_type}, status={ocr.status}"
    )

    try:
        cached = get_ocr_result(source_type, file_hash)
        if cached and cached.get("text") is not None:
            text = cached.get("text", "")
            print(f"[OCR] Redis cache hit: source_type={source_type}, hash={file_hash[:12]}")
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
        print(f"[OCR] OCRJob {ocr.ocr_id} DONE, text_len={len(text)}")
    except Exception as e:
        ocr.status = "FAILED"
        ocr.text = f"OCR ERROR: {e}"
        print(f"[OCR][ERROR] OCRJob {ocr.ocr_id} FAILED: {e}")

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

    print(
        f"[OCR] OCRJob final: ocr_id={ocr.ocr_id}, status={ocr.status}, "
        f"completed_at={ocr.completed_at}"
    )
    return ocr


# ==================================================
# OCR raw text → Visit 구조화
# ==================================================
def parse_ocr_text_to_visit(text: str) -> VisitFormSchema:
    """
    OCR result text → Visit Form 자동 구조화
    GPT가 camelCase / 기타 키로 줘도 스키마에 맞게 매핑
    """
    print("\n[PARSE][VISIT] ===== parse_ocr_text_to_visit START =====")
    print(f"[PARSE][VISIT] input text length: {len(text)}")
    print("[PARSE][VISIT] input head:")
    print(text[:500])

    cached = get_parse_result("visit", text)
    if cached:
        print("[PARSE][VISIT] Redis cache hit")
        return VisitFormSchema(**cached)

    try:
        raw = parse_visit_form_from_ocr(text) or {}
        print("[PARSE][VISIT] raw dict from GPT:")
        print(raw)

        data: dict[str, str] = {}

        # 병원명
        data["hospital"] = (
            raw.get("hospital")
            or raw.get("hospital_name")
            or raw.get("clinic")
            or ""
        )

        # 의사 이름
        data["doctor_name"] = (
            raw.get("doctor_name")
            or raw.get("doctor")
            or raw.get("physician")
            or ""
        )

        # 증상
        data["symptom"] = (
            raw.get("symptom")
            or raw.get("symptoms")
            or raw.get("chief_complaint")
            or ""
        )

        # 소견/메모
        data["opinion"] = (
            raw.get("opinion")
            or raw.get("notes")
            or raw.get("assessment")
            or ""
        )

        # 진단 코드 & 이름
        data["diagnosis_code"] = (
            raw.get("diagnosis_code")
            or raw.get("diagnosisCode")
            or raw.get("icd_code")
            or ""
        )
        data["diagnosis_name"] = (
            raw.get("diagnosis_name")
            or raw.get("diagnosisName")
            or raw.get("diagnosis")
            or ""
        )

        # 날짜
        data["date"] = (
            raw.get("date")
            or raw.get("visit_date")
            or raw.get("visitDate")
            or str(datetime.today().date())
        )

        print("[PARSE][VISIT] mapped data:")
        print(data)
        print("[PARSE][VISIT] ===== SUCCESS =====\n")
        parsed = VisitFormSchema(**data)
        set_parse_result("visit", text, parsed.model_dump())
        return parsed

    except Exception as e:
        print(f"[PARSE][VISIT][ERROR] parsing failed: {e}")
        dummy = {
            "hospital": "",
            "doctor_name": "",
            "symptom": text[:200],
            "opinion": "",
            "diagnosis_code": "",
            "diagnosis_name": "",
            "date": str(datetime.today().date()),
        }
        print("[PARSE][VISIT] returning dummy data:")
        print(dummy)
        print("[PARSE][VISIT] ===== FALLBACK =====\n")
        parsed = VisitFormSchema(**dummy)
        set_parse_result("visit", text, parsed.model_dump())
        return parsed


# ==================================================
# OCR raw text → Prescription 구조화 (여러 약)
# ==================================================
def parse_ocr_text_to_prescription(text: str) -> List[PrescriptionFormSchema]:
    """
    OCR result text → 여러 개의 Prescription Form 자동 구조화
    GPT는 { "medications": [ {...}, {...}, ... ] } 형태로 응답하고,
    여기서는 그 배열을 순회하며 PrescriptionFormSchema 리스트로 변환한다.
    """
    print("\n[PARSE][PRESC] ===== parse_ocr_text_to_prescription START =====")
    print(f"[PARSE][PRESC] input text length: {len(text)}")
    print("[PARSE][PRESC] input head:")
    print(text[:500])

    cached = get_parse_result("prescription", text)
    if cached:
        print("[PARSE][PRESC] Redis cache hit")
        return [PrescriptionFormSchema(**item) for item in cached.get("items", [])]

    try:
        raw = parse_prescription_form_from_ocr(text) or {}
        print("[PARSE][PRESC] raw dict from GPT:")
        print(raw)

        meds_raw = raw.get("medications", [])

        # 방어 코드: dict 로 온 경우 / 아예 리스트가 아닌 경우 처리
        if isinstance(meds_raw, dict):
            meds_list = [meds_raw]
        elif isinstance(meds_raw, list):
            meds_list = meds_raw
        else:
            # 혹시 GPT가 옛 형식(단일 객체)으로 준 경우 대비
            meds_list = [raw]

        parsed_list: List[PrescriptionFormSchema] = []

        for idx, m in enumerate(meds_list):
            if not isinstance(m, dict):
                print(f"[PARSE][PRESC] skip non-dict medication at index {idx}: {m}")
                continue

            data: dict[str, object] = {}

            # 약 이름
            data["med_name"] = (
                m.get("med_name")
                or m.get("medName")
                or m.get("drug_name")
                or m.get("name")
                or ""
            )

            # 제형
            data["dosage_form"] = (
                m.get("dosage_form")
                or m.get("dosageForm")
                or m.get("form")
                or ""
            )

            # 용량 / 단위
            data["dose"] = (
                m.get("dose")
                or m.get("dose_amount")
                or m.get("strength")
                or ""
            )
            data["unit"] = m.get("unit") or m.get("dose_unit") or ""

            # 복용 시간(schedule)
            schedule = (
                m.get("schedule")
                or m.get("dose_times")
                or m.get("when")
                or []
            )

            if isinstance(schedule, str):
                items = [s.strip() for s in schedule.split(",") if s.strip()]
                data["schedule"] = items
            elif isinstance(schedule, list):
                data["schedule"] = [
                    str(s).strip() for s in schedule if str(s).strip()
                ]
            else:
                data["schedule"] = []

            # 기타 복약 시간
            data["custom_schedule"] = (
                m.get("custom_schedule")
                or m.get("customSchedule")
                or m.get("etc")
                or None
            )

            # 시작일 / 종료일
            data["start_date"] = (
                m.get("start_date") or m.get("startDate") or ""
            )
            data["end_date"] = m.get("end_date") or m.get("endDate") or ""

            print(f"[PARSE][PRESC] mapped data (index {idx}):")
            print(data)

            try:
                parsed_list.append(PrescriptionFormSchema(**data))
            except Exception as e:
                print(f"[PARSE][PRESC] Pydantic validation failed at index {idx}: {e}")

        if not parsed_list:
            # 아무 것도 못 만들었다면 dummy 하나라도 리턴
            dummy = PrescriptionFormSchema(
                med_name="",
                dosage_form="",
                dose="",
                unit="",
                schedule=[],
                custom_schedule=None,
                start_date="",
                end_date="",
            )
            print("[PARSE][PRESC] no valid meds, returning single dummy")
            print(dummy)
            print("[PARSE][PRESC] ===== FALLBACK (EMPTY) =====\n")
            set_parse_result("prescription", text, {"items": [dummy.model_dump()]})
            return [dummy]

        print(f"[PARSE][PRESC] total parsed meds: {len(parsed_list)}")
        print("[PARSE][PRESC] ===== SUCCESS =====\n")
        set_parse_result(
            "prescription",
            text,
            {"items": [item.model_dump() for item in parsed_list]},
        )
        return parsed_list

    except Exception as e:
        print(f"[PARSE][PRESC][ERROR] parsing failed: {e}")
        dummy = PrescriptionFormSchema(
            med_name="",
            dosage_form="",
            dose="",
            unit="",
            schedule=[],
            custom_schedule=None,
            start_date="",
            end_date="",
        )
        print("[PARSE][PRESC] returning dummy list:")
        print(dummy)
        print("[PARSE][PRESC] ===== FALLBACK (EXCEPTION) =====\n")
        set_parse_result("prescription", text, {"items": [dummy.model_dump()]})
        return [dummy]
