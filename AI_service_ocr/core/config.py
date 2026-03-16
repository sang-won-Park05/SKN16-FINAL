# core/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# 업로드 경로
OCR_UPLOAD_DIR = os.getenv("OCR_UPLOAD_DIR", "uploads/ocr")

# GPT 모델
OCR_GPT_MODEL = os.getenv("OCR_GPT_MODEL", "gpt-4o-mini")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OCR_REDIS_TTL = int(os.getenv("OCR_REDIS_TTL", "86400"))
OCR_PARSE_REDIS_TTL = int(os.getenv("OCR_PARSE_REDIS_TTL", "86400"))
