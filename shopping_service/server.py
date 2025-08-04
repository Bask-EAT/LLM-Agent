from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import json
from agent import shopping_agent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ShoppingAgent Server", description="텍스트 기반 레시피 검색 서버")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShoppingRequest(BaseModel):
    message: str

class ShoppingResponse(BaseModel):
    answer: str
    ingredients: list
    recipe: list
    agent_type: str = "shopping"

@app.post("/chat", response_model=ShoppingResponse)
async def chat_message(request: Request):
    """채팅 메시지 처리 (프론트엔드 호환용)"""
    try:
        # 들어오는 데이터 로깅
        logger.info(f"=== /chat 엔드포인트 호출됨 ===")
        logger.info(f"요청 헤더: {dict(request.headers)}")
        
        # JSON 데이터 직접 받기
        body = await request.json()
        logger.info(f"받은 JSON 데이터: {body}")
        
        message = body.get("message")
        if not message:
            logger.error("message 필드가 없습니다")
            raise HTTPException(status_code=400, detail="message 필드가 필요합니다.")
        
        logger.info(f"처리할 메시지: {message}")
        
        # ShoppingAgent로 메시지 처리
        result = await shopping_agent.process_message(message)
        logger.info(f"ShoppingAgent 처리 결과: {result}")
        
        return ShoppingResponse(**result)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise HTTPException(status_code=422, detail="잘못된 JSON 형식입니다.")
    except Exception as e:
        logger.error(f"채팅 메시지 처리 오류: {e}")
        raise HTTPException(status_code=500, detail="메시지 처리 중 오류가 발생했습니다.")

@app.post("/process", response_model=ShoppingResponse)
async def process_message(request: ShoppingRequest):
    """텍스트 기반 레시피 검색 처리"""
    try:
        logger.info(f"=== /process 엔드포인트 호출됨 ===")
        logger.info(f"처리할 메시지: {request.message}")
        
        result = await shopping_agent.process_message(request.message)
        logger.info(f"ShoppingAgent 처리 결과: {result}")
        
        return ShoppingResponse(**result)
    except Exception as e:
        logger.error(f"ShoppingAgent 처리 오류: {e}")
        raise HTTPException(status_code=500, detail="레시피 검색 중 오류가 발생했습니다.")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "ShoppingAgent Server"}

@app.get("/")
async def root():
    """루트 엔드포인트"""
    logger.info("=== 루트 엔드포인트 호출됨 ===")
    return {
        "message": "ShoppingAgent Server is running",
        "endpoints": {
            "/chat": "POST - 채팅 메시지 처리",
            "/process": "POST - 레시피 검색 처리", 
            "/health": "GET - 서버 상태 확인"
        }
    }

if __name__ == "__main__":
    logger.info("=== ShoppingAgent Server 시작 ===")
    uvicorn.run(app, host="0.0.0.0", port=8002) 