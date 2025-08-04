from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
import json
import os

# config 모듈 import (Google Cloud 인증 설정을 위해)
import config

# core 모듈에서 함수 import
from core.extractor import process_video_url

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VideoAgent Server", description="유튜브 영상 레시피 추출 서버")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    youtube_url: str
    message: str

class VideoResponse(BaseModel):
    answer: str
    ingredients: list
    recipe: list
    agent_type: str = "video"

@app.post("/process", response_model=VideoResponse)
async def process_video(request: Request):
    """유튜브 영상 레시피 추출 처리"""
    try:
        # 들어오는 데이터 로깅
        logger.info(f"=== /process 엔드포인트 호출됨 ===")
        logger.info(f"요청 헤더: {dict(request.headers)}")
        
        # JSON 데이터 직접 받기
        body = await request.json()
        logger.info(f"받은 JSON 데이터: {body}")
        
        youtube_url = body.get("youtube_url") or body.get("message")
        if not youtube_url:
            logger.error("youtube_url 또는 message 필드가 없습니다")
            raise HTTPException(status_code=400, detail="youtube_url 또는 message 필드가 필요합니다.")
        
        logger.info(f"처리할 유튜브 URL: {youtube_url}")
        
        # VideoAgent로 영상 처리
        result = process_video_url(youtube_url)
        logger.info(f"VideoAgent 처리 결과: {result}")
        
        return VideoResponse(**result)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise HTTPException(status_code=422, detail="잘못된 JSON 형식입니다.")
    except Exception as e:
        logger.error(f"영상 처리 오류: {e}")
        raise HTTPException(status_code=500, detail="영상 처리 중 오류가 발생했습니다.")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "VideoAgent Server"}

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "VideoAgent Server is running",
        "endpoints": {
            "/process": "POST - 유튜브 영상 레시피 추출",
            "/health": "GET - 서버 상태 확인"
        }
    }

if __name__ == "__main__":
    logger.info("=== VideoAgent Server 시작 ===")
    # 유튜브 영상 처리는 시간이 오래 걸리므로 타임아웃을 늘림
    uvicorn.run(app, host="0.0.0.0", port=8003, timeout_keep_alive=600)