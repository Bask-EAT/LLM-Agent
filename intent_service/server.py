from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse
import aiohttp
from planning_agent import run_agent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Planning Agent Server", description="모든 사용자 요청을 처리하는 메인 서버")

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

# youtube_url을 감지하는 간단한 함수
def is_youtube_url_request(message: str) -> bool:
    return "youtube.com" in message or "youtu.be" in message


@app.post("/chat")
async def chat_with_agent(request: Request):
    """사용자 입력의 의도를 분류"""
    try:
        # # 들어오는 데이터 로깅
        # logger.info(f"요청 헤더: {dict(request.headers)}")
        
        # # JSON 데이터 직접 받기
        # body = await request.json()
        # logger.info(f"받은 JSON 데이터: {body}")
        
        # # youtube_url 또는 message 중 하나를 사용
        # # user_message = body.get("youtube_url") or body.get("message")
        # user_message = body.get("message")
        # logger.info(f"추출된 메시지: {user_message}")
        
        # if not user_message:
        #     raise HTTPException(status_code=400, detail="message 또는 youtube_url이 필요합니다.")
        

        # if is_youtube_url_request(user_message):
        #     # 유튜브 URL이 감지되면 VideoAgent Service로 요청 전달
        #     response_data = await forward_to_video_service(user_message)    # JSON 객체 반환
        #     logger.info(f"VideoAgent Service 직접 호출 결과: {response_data}")
        #     return {"response": response_data}      # 프런트엔드에 JSON 객체를 그대로 반환
        # else:
        #     # 유튜브 URL이 없으면 기존처럼 에이전트를 실행
        #     response_json = await run_agent(user_message)
        #     logger.info(f"에이전트 응답: {response_json}")
            
        #     return {"response": response_json}
        body = await request.json()
        user_message = body.get("message")

        if not user_message:
             return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "detail": "message가 필요합니다."}
            )
        # 모든 입력을 Agent에 전달 (Agent가 URL 포함 여부를 판단하고 도구 순서를 결정)
        response_data = await run_agent(user_message)
        logger.info(f"---------에이전트 응답: {response_data}")

        return JSONResponse(content={"response": response_data})


    except Exception as e:
        logger.error(f"에이전트 처리 중 오류: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": f"서버 내부 오류가 발생했습니다: {e}"}
        )


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