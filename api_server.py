from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat_core import get_openai_client, search_all_content, get_agent_system_prompt

app = FastAPI(title="K-ICIS VOC API", version="1.0.0")

# CORS 설정 - smartChatAssist에서 호출할 수 있도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    content: str
    isUser: bool

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "K-ICIS VOC API Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "K-ICIS VOC API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    K-ICIS VOC 전문 상담 챗봇 API
    """
    try:
        # 대화 이력을 OpenAI 형식으로 변환
        messages = []
        
        # VOC 에이전트 시스템 프롬프트 추가
        search_result = search_all_content(request.message, pdf_top_k=8, agent_type='voc_agent')
        
        if search_result['context_text']:
            agent_system_prompt = get_agent_system_prompt('voc_agent', search_result['context_text'])
        else:
            agent_system_prompt = get_agent_system_prompt('voc_agent', "")
        
        messages.append({"role": "system", "content": agent_system_prompt})
        
        # 대화 이력 추가
        for msg in request.conversation_history:
            role = "user" if msg.isUser else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        # 현재 사용자 메시지 추가
        messages.append({"role": "user", "content": request.message})
        
        # OpenAI API 호출
        response = get_openai_client(messages)
        
        return ChatResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        print(f"Chat API 오류: {e}")
        return ChatResponse(
            response="죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            success=False,
            error=str(e)
        )

@app.post("/api/voc/chat")
async def voc_chat_endpoint(request: ChatRequest):
    """
    VOC 전용 채팅 엔드포인트 (별칭)
    """
    return await chat_endpoint(request)

if __name__ == "__main__":
    print("🚀 K-ICIS VOC API 서버 시작...")
    print("📍 URL: http://localhost:8001")
    print("📋 문서: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
