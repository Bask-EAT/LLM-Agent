from fastapi import FastAPI, HTTPException, Request, BackgroundTasks 
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse
import asyncio
import aiohttp
from planning_agent import run_agent
import uuid
import time

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
# TextAgent Service URL
TEXT_SERVICE_URL = "http://localhost:8002"

# 간단 상태 저장소 (세션 관리 미구현 환경에서 최근 결과를 보관)
recent_results = {
    "text": None,
    "video": None,
    "first_source": None,
}

def is_youtube_url_request(message: str) -> bool:
    if not message:
        return False
    lower = message.lower()
    return ("youtube.com" in lower) or ("youtu.be" in lower)

def normalize_ingredient_string(raw: str):
    """문자열 재료를 {item, amount, unit}로 최대한 보수적으로 정규화"""
    import re
    text = (raw or "").strip()
    if not text:
        return {"item": "", "amount": "", "unit": ""}

    # 1) 콜론 구분: "식용유: 5큰술" / "소고기: 200 g"
    m_colon = re.match(r"^(.+?)\s*[:：]\s*(.+)$", text)
    if m_colon:
        item = m_colon.group(1).strip()
        rhs = m_colon.group(2).strip()
        # 숫자+단위 붙어있는 형태 포함: 5큰술, 200g, 1/4통
        m_q = re.match(r"^(\d+[\./,]?\d*)\s*([가-힣A-Za-z%]+)$", rhs)
        if m_q:
            return {"item": item, "amount": m_q.group(1), "unit": m_q.group(2)}
        # 약간/적당량 등
        return {"item": item, "amount": rhs, "unit": ""}

    # 2) 괄호 수량: "올리브유 (3큰술)"
    m_paren = re.match(r"^(.+?)\s*\(([^)]+)\)$", text)
    if m_paren:
        item = m_paren.group(1).strip()
        qty = m_paren.group(2).strip()
        m_q = re.match(r"^(\d+[\./,]?\d*)\s*([가-힣A-Za-z%]+)$", qty)
        if m_q:
            return {"item": item, "amount": m_q.group(1), "unit": m_q.group(2)}
        return {"item": item, "amount": qty, "unit": ""}

    # 3) 공백 구분: "새우 10 마리" 또는 붙은 단위: "10마리", "200g", "1/4통"
    m_space = re.match(r"^([가-힣A-Za-z\s]+?)\s*(\d+[\./,]?\d*)\s*([가-힣A-Za-z%]+)?$", text)
    if m_space:
        item = m_space.group(1).strip()
        amount = (m_space.group(2) or "").strip()
        unit = (m_space.group(3) or "").strip()
        return {"item": item, "amount": amount, "unit": unit}

    # 4) 단독 항목
    return {"item": text, "amount": "", "unit": ""}

def build_recipe_object(source: str, payload: dict) -> dict:
    food_name = payload.get("food_name") or payload.get("title") or ""
    ingredients_raw = payload.get("ingredients", [])
    recipe_steps = payload.get("recipe") or payload.get("steps") or []
    if isinstance(ingredients_raw, list):
        structured = []
        for ing in ingredients_raw:
            if isinstance(ing, dict) and {"item", "amount", "unit"}.issubset(ing.keys()):
                structured.append({
                    "item": str(ing.get("item", "")),
                    "amount": str(ing.get("amount", "")),
                    "unit": str(ing.get("unit", ""))
                })
            elif isinstance(ing, str):
                structured.append(normalize_ingredient_string(ing))
        ingredients = structured
    else:
        ingredients = []
    return {
        "source": source,
        "food_name": food_name,
        "ingredients": ingredients,
        "recipe": recipe_steps if isinstance(recipe_steps, list) else []
    }

def extract_dish_names(message: str) -> list:
    """메시지에서 여러 요리명을 간단 규칙으로 분리 추출"""
    import re
    if not message:
        return []
    text = message.strip()
    # URL 제거
    text = re.sub(r"https?://\S+", " ", text)
    # 구분자 통일 (와/과/랑/및/그리고/,+/ 등)
    text = re.sub(r"\s*(와|과|랑|및|그리고|,|/|\+)\s*", ",", text)
    # 잡어 제거
    text = re.sub(r"(레시피|조리법|만드는\s*법|알려줘|주세요|좀)", "", text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    # 필터링 및 중복 제거
    seen = set()
    results = []
    for p in parts:
        if len(p) < 2 or len(p) > 25:
            continue
        if p in seen:
            continue
        seen.add(p)
        results.append(p)
    return results

def extract_requested_count(message: str) -> int:
    """문장에서 요청 개수(N)를 추출. '3개', '세 가지' 등 지원"""
    import re
    if not message:
        return 0
    text = message.strip()
    # 숫자 기반: 3개, 2가지 등
    m = re.search(r"(\d+)\s*(개|가지)", text)
    if m:
        try:
            return max(0, int(m.group(1)))
        except Exception:
            pass
    # 한글 수사
    num_map = {
        "한": 1, "두": 2, "세": 3, "네": 4,
        "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10
    }
    for word, val in num_map.items():
        if re.search(fr"{word}\s*(개|가지)", text):
            return val
    return 0

def detect_category(message: str) -> str:
    """간단 카테고리 감지. 기본 한식"""
    lower = (message or "").lower()
    if any(k in message or k in lower for k in ["한식", "korean", "코리안"]):
        return "한식"
    if any(k in message or k in lower for k in ["중식", "중국", "차이니즈", "chinese"]):
        return "중식"
    if any(k in message or k in lower for k in ["일식", "일본", "japanese", "japan"]):
        return "일식"
    if any(k in message or k in lower for k in ["이탈리아", "이탈리아식", "italian", "파스타"]):
        return "이탈리아식"
    if any(k in message or k in lower for k in ["미국", "미국식", "american", "버거"]):
        return "미국식"
    return "한식"

# youtube_url을 감지하는 간단한 함수
def is_youtube_url_request(message: str) -> bool:
    return "youtube.com" in message or "youtu.be" in message


# 작업 상태와 결과를 저장할 인메모리 딕셔너리
# (서버 재시작 시 초기화됨. 영구 보관이 필요하면 Redis나 DB 사용)
jobs = {}

async def run_agent_and_store_result(job_id: str, user_message: str):
    """
    백그라운드에서 에이전트를 실행하고 결과를 jobs 딕셔너리에 저장하는 함수
    """
    logger.info(f"=== 🤍Background-Task-{job_id}: 작업 시작. ===")
    jobs[job_id] = {"status": "processing", "start_time": time.time()}
    try:
        result = await run_agent(user_message)
        logger.info(f"=== 🤍 Agent 최종 응답: {result} 🤍 ===")
        jobs[job_id] = {"status": "completed", "result": result}
        logger.info(f"=== 🤍Background-Task-{job_id}: 작업 완료. ===")
    except Exception as e:
        logger.error(f"=== 🤍Background-Task-{job_id}: 작업 중 에러 발생: {e}", exc_info=True)
        jobs[job_id] = {"status": "failed", "error": str(e)}


# @app.post("/chat")
# async def chat_with_agent(request: Request):
#     """사용자 요청을 받아 텍스트/비디오 결과를 통합 스키마로 반환"""
#     try:
#         body = await request.json()
#         logger.info(f"=== 🤍intent_service에서 /chat 엔드포인트 호출됨🤍 ===")
#         user_message = body.get("message")
#         logger.info(f"=== 🤍사용자 메시지: {user_message}")
#         if not user_message:
#             return JSONResponse(status_code=400, content={"error": "Bad Request", "detail": "message가 필요합니다."})

#         agent_response = await run_agent(user_message)

#         logger.info(f"=== 🤍 Agent 최종 응답: {agent_response} 🤍 ===")
        
#         # Agent가 생성한 JSON 응답을 그대로 클라이언트에게 전달합니다.
#         return JSONResponse(content={"response": agent_response})


#     except Exception as e:
#         logger.error(f"에이전트 처리 중 오류: {e}", exc_info=True)
#         return JSONResponse(
#             status_code=500,
#             content={"error": "Internal Server Error", "detail": f"서버 내부 오류가 발생했습니다: {e}"}
#         )


# 즉시 job_id를 반환.
@app.post("/chat")
async def chat_with_agent(request: Request, background_tasks: BackgroundTasks):
    """
    사용자 요청을 받아 작업을 백그라운드에 등록하고 즉시 작업 ID를 반환합니다.
    """
    try:
        body = await request.json()
        logger.info(f"=== 🤍intent_service에서 /chat 엔드포인트 호출됨🤍 ===")
        user_message = body.get("message")
        logger.info(f"=== 🤍사용자 메시지: {user_message}")
        if not user_message:
            raise HTTPException(status_code=400, detail="message가 필요합니다.")

        job_id = str(uuid.uuid4()) # 고유한 작업 ID 생성
        
        # 백그라운드에서 run_agent_and_store_result 함수를 실행하도록 등록
        background_tasks.add_task(run_agent_and_store_result, job_id, user_message)
        
        # 클라이언트에게는 작업 ID를 즉시 반환
        return JSONResponse(status_code=202, content={"job_id": job_id})
        
    except Exception as e:
        logger.error(f"에이전트 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다: {e}")


# 작업 상태를 알려줌.
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    주어진 작업 ID의 상태와 결과를 반환합니다.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=job)



async def forward_to_video_service(youtube_url: str):
    """VideoAgent Service로 유튜브 링크 전달"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "youtube_url": youtube_url,
                "message": youtube_url
            }
            logger.debug("=== 🤍payload for VideoAgent Service: %s", payload)
            
            logger.info(f"=== 🤍VideoAgent Service로 요청 전송: {VIDEO_SERVICE_URL}/process")
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

async def forward_to_text_service(message: str):
    """TextAgent Service로 텍스트 질의 전달"""
    # try:
    #     async with aiohttp.ClientSession() as session:
    #         payload = {"message": message}
    #         logger.debug("=== 🤍payload for TextAgent Service: %s", payload)

    #         logger.info(f"=== 🤍TextAgent Service로 요청 전송: {TEXT_SERVICE_URL}/process")
    #         async with session.post(f"{TEXT_SERVICE_URL}/process", json=payload) as response:
    #             if response.status == 200:
    #                 result = await response.json()
    #                 logger.info(f"TextAgent Service 응답: {result}")
    #                 return result
    #             else:
    #                 error_text = await response.text()
    #                 logger.error(f"TextAgent Service 오류 (상태: {response.status}): {error_text}")
    #                 return {
    #                     "error": f"TextAgent Service 오류: {response.status}",
    #                     "message": error_text
    #                 }
    # except aiohttp.ClientConnectorError as e:
    #     logger.error(f"TextAgent Service 연결 실패: {e}")
    #     return {
    #         "error": "TextAgent Service에 연결할 수 없습니다.",
    #         "message": "8002 서버가 실행 중인지 확인해주세요."
    #     }
    # except Exception as e:
    #     logger.error(f"TextAgent Service 호출 중 오류: {e}")
    #     return {
    #         "error": "TextAgent Service 호출 중 오류가 발생했습니다.",
    #         "message": str(e)
    #     }
    # except Exception as e:
    #     logger.error(f"TextAgent Service 호출 중 오류: {e}")
    #     return {
    #         "error": "TextAgent Service 호출 중 오류가 발생했습니다.",
    #         "message": str(e)
    #     }

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "Intent LLM Server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 