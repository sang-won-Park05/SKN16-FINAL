"""
STT API Server (port 8002)
메인 레포(port 8000)에서 HTTP 요청을 받아 STT 처리하는 독립 API 서버
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router, start_stt_worker, stop_stt_worker
from core.crud import init_db
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(
    title="STT Processing API",
    description="Speech-to-Text processing service with OpenAI Whisper",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STT 라우터 등록
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 DB 초기화"""
    init_db()
    start_stt_worker()
    print("✅ STT API Server started on port 8002")


@app.on_event("shutdown")
async def shutdown_event():
    stop_stt_worker()


@app.get("/")
async def root():
    """Health check"""
    return {
        "message": "STT Processing API is running",
        "version": "1.0.0",
        "port": 8002
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
