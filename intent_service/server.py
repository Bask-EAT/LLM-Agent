from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
import aiohttp
from classifier import intent_classifier

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intent LLM Server", description="사용자 입력 의도 분류 서버")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# VideoAgent Service URL
VIDEO_SERVICE_URL = "http://localhost:8003"

@app.post("/classify")
async def classify_intent(request: Request):
    """사용자 입력의 의도를 분류"""
    try:
        # 들어오는 데이터 로깅
        logger.info(f"요청 헤더: {dict(request.headers)}")
        
        # JSON 데이터 직접 받기
        body = await request.json()
        logger.info(f"받은 JSON 데이터: {body}")
        
        # youtube_url 또는 message 중 하나를 사용
        user_message = body.get("youtube_url") or body.get("message")
        logger.info(f"추출된 메시지: {user_message}")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="message 또는 youtube_url이 필요합니다.")
        
        result = await intent_classifier.classify_intent(user_message)
        logger.info(f"classifier 결과: {result}")
        
        # 유튜브 링크가 감지된 경우 8003 서버로 전달
        if result.get("intent") == "VIDEO":
            logger.info("유튜브 링크 감지됨. VideoAgent Service로 전달합니다.")
            video_result = await forward_to_video_service(user_message)
            return {
                "intent": "VIDEO",
                "confidence": result.get("confidence", 0.95),
                "reason": "유튜브 링크가 감지되어 VideoAgent Service에서 처리되었습니다.",
                "message": user_message,
                "video_result": video_result
            }
        
        logger.info(f"응답 반환 타입: {type(result)}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise HTTPException(status_code=422, detail="잘못된 JSON 형식입니다.")
    except Exception as e:
        logger.error(f"의도 분류 오류: {e}")
        raise HTTPException(status_code=500, detail="의도 분류 중 오류가 발생했습니다.")

async def forward_to_video_service(youtube_url: str):
    """VideoAgent Service로 유튜브 링크 전달"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "youtube_url": youtube_url,
                "message": youtube_url
            }
            
            logger.info(f"VideoAgent Service로 요청 전송: {VIDEO_SERVICE_URL}/process")
            async with session.post(f"{VIDEO_SERVICE_URL}/process", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"VideoAgent Service 응답: {result}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"VideoAgent Service 오류 (상태: {response.status}): {error_text}")
                    return {
                        "error": f"VideoAgent Service 오류: {response.status}",
                        "message": error_text
                    }
    except aiohttp.ClientConnectorError as e:
        logger.error(f"VideoAgent Service 연결 실패: {e}")
        return {
            "error": "VideoAgent Service에 연결할 수 없습니다.",
            "message": "8003 서버가 실행 중인지 확인해주세요."
        }
    except Exception as e:
        logger.error(f"VideoAgent Service 호출 중 오류: {e}")
        return {
            "error": "VideoAgent Service 호출 중 오류가 발생했습니다.",
            "message": str(e)
        }

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "Intent LLM Server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 