# AI_service_LLM/chatbot/core/llm.py

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from openai import OpenAI

# ============================================
# 🔹 OpenAI 클라이언트 & 기본 모델 설정
# ============================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "gpt-4o-mini")

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ============================================
# 🔹 공통 LLM 호출 함수
# ============================================

def call_llm(
    system_prompt: str,
    user_message: str,
    context: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    에이전트에서 공통으로 사용하는 LLM 호출 함수.

    - system_prompt: 시스템 역할 설명 (에이전트별 프롬프트)
    - user_message: 사용자의 실제 질문
    - context: RAG로 검색된 문서들 (선택)
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if context:
        messages.append(
            {
                "role": "user",
                "content": f"[검색된 참고 정보]\n{context}",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    client = get_client()
    resp = client.chat.completions.create(
        model=model or CHATBOT_MODEL,
        messages=messages,
        temperature=temperature,
    )

    return resp.choices[0].message.content or ""


# run_llm 이라는 이름을 쓰는 코드도 있을 수 있으니 alias 제공
def run_llm(
    system_prompt: str,
    user_message: str,
    context: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    return call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        context=context,
        model=model,
        temperature=temperature,
    )


# ============================================
# 🔹 Embedding 생성 함수
# ============================================

def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """
    OpenAI 임베딩 모델을 사용하여 텍스트의 임베딩 벡터를 반환합니다.
    """
    client = get_client()
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

