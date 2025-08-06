from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import json
from agent import shopping_agent
import sys, os

# --- 다른 폴더에 있는 모듈을 가져오기 위한 경로 설정 ---
# 현재 파일(planning_agent.py)이 있는 디렉토리의 절대 경로를 가져옵니다.
# 예: /path/to/your/project/intent_service
current_dir = os.path.dirname(os.path.abspath(__file__))

# 부모 디렉토리(프로젝트의 루트 폴더) 경로를 가져옵니다.
# 예: /path/to/your/project
project_root = os.path.dirname(current_dir)

# 파이썬이 모듈을 검색하는 경로 리스트에 프로젝트 루트 폴더를 추가합니다.
sys.path.append(project_root)
# ----------------------------------------------------
from intent_service.planning_agent import run_agent


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

@app.post("/chat", response_model=ShoppingResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(request: Request):
    """    
    사용자의 모든 메시지를 받아 플래닝 에이전트가 처리합니다.
    세션별로 대화 기록을 관리합니다.
    """
    try:
        # 들어오는 데이터 로깅
        logger.info(f"=== /chat 엔드포인트 호출됨 ===")
        logger.info(f"요청 헤더: {dict(request.headers)}")
        
        # JSON 데이터 직접 받기
        body = await request.json()
        logger.info(f"받은 JSON 데이터: {body}")
        
        # session_id = body.get("session_id", "default_session") # 클라이언트에서 세션 ID를 보내주면 좋습니다.

        message = body.get("message")
        if not message:
            logger.error("message 필드가 없습니다")
            raise HTTPException(status_code=400, detail="message 필드가 필요합니다.")
        
        logger.info(f"처리할 메시지: {message}")
        
        # 에이전트 실행 (session_id 추가)
        session_id = "default_session"  # 임시로 기본 세션 ID 사용
        response_text = await run_agent(message, session_id)
        
        logger.info(f"최종 응답: {response_text}")
        
        # 프론트엔드가 기대하는 형식으로 응답 변환
        response_data = {
            "answer": response_text,
            "ingredients": [],  # 일단 빈 배열로 설정
            "recipe": []       # 일단 빈 배열로 설정
        }
        logger.info(f"반환할 응답 데이터: {response_data}")
        
        return response_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise HTTPException(status_code=422, detail="잘못된 JSON 형식입니다.")
    except Exception as e:
        logger.error(f"에이전트 처리 중 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다: {e}")

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
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True) 