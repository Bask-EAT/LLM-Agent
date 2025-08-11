from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse
import asyncio
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


@app.post("/chat")
async def chat_with_agent(request: Request):
    """사용자 요청을 받아 텍스트/비디오 결과를 통합 스키마로 반환"""
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
        

        body = await request.json()
        user_message = body.get("message")
        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Bad Request", "detail": "message가 필요합니다."})

        current_source = "video" if is_youtube_url_request(user_message) else "text"

        recipes = []
        answer = "요청하신 레시피를 정리했습니다."
        if current_source == "video":
            # 두 작업을 동시에 실행하여 전체 응답 시간을 단축
            try:
                video_task = asyncio.wait_for(forward_to_video_service(user_message), timeout=180)
                text_task = asyncio.wait_for(forward_to_text_service(user_message), timeout=180)
                video_res, text_res = await asyncio.gather(video_task, text_task, return_exceptions=True)

                if isinstance(video_res, dict):
                    recipe_video = build_recipe_object("video", video_res)
                    recent_results["video"] = recipe_video
                    if recent_results["first_source"] is None:
                        recent_results["first_source"] = "video"
                    recipes.append(recipe_video)
                    answer = video_res.get("answer", answer)
                else:
                    logger.error(f"video 작업 실패: {video_res}")

                if isinstance(text_res, dict):
                    recipe_text = build_recipe_object("text", text_res)
                    recent_results["text"] = recipe_text
                    recipes.append(recipe_text)
                else:
                    logger.error(f"text 작업 실패: {text_res}")
            except Exception as e:
                logger.error(f"동시 실행 처리 중 오류: {e}")
        else:
            # 텍스트 질문: 다요리 분리 + N개 개수 인식 후 병렬 호출
            try:
                dish_names = extract_dish_names(user_message)
                requested_n = extract_requested_count(user_message)
                targets = []
                if dish_names:
                    targets = [f"{name} 레시피" for name in dish_names]
                # 요청 개수가 명시되고 타겟이 부족하면 카테고리 기반으로 보충
                if requested_n and len(targets) < requested_n:
                    category = detect_category(user_message)
                    defaults_by_cat = {
                        "한식": ["김치찌개", "된장찌개", "비빔밥", "불고기", "잡채"],
                        "중식": ["짜장면", "짬뽕", "마파두부", "꿔바로우"],
                        "일식": ["규동", "가츠동", "우동", "야키소바"],
                        "이탈리아식": ["알리오 올리오", "까르보나라", "마르게리타 피자"],
                        "미국식": ["치즈버거", "맥앤치즈", "프라이드치킨"]
                    }
                    pool = defaults_by_cat.get(category, defaults_by_cat["한식"]) 
                    for nm in pool:
                        if len(targets) >= requested_n:
                            break
                        if nm not in [t.replace(" 레시피", "") for t in targets]:
                            targets.append(f"{nm} 레시피")

                if not targets:
                    targets = [f"{user_message} 레시피"]

                if requested_n:
                    targets = targets[:requested_n]

                tasks = [asyncio.wait_for(forward_to_text_service(name), timeout=120) for name in targets]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, dict):
                        recipe_text = build_recipe_object("text", res)
                        recent_results["text"] = recipe_text
                        recipes.append(recipe_text)
                        if recent_results["first_source"] is None:
                            recent_results["first_source"] = "text"
                        if res.get("answer"):
                            answer = res.get("answer")
                    else:
                        logger.error(f"text 병렬 작업 실패: {res}")
            except Exception as e:
                logger.error(f"text 병렬 처리 중 오류: {e}")

        # 2) 통합 응답 조립 (첫 사용 소스가 맨 앞)
        if not recipes:
            first = recent_results.get("first_source")
            if first and recent_results.get(first):
                recipes.append(recent_results[first])
            other = "video" if first == "text" else "text"
            if recent_results.get(other):
                recipes.append(recent_results[other])

        # 둘 다 실패했으면 500 반환
        if not recipes:
            return JSONResponse(status_code=500, content={"error": "No recipes", "detail": "두 소스 모두 처리에 실패했습니다."})

        # 중복 제거: source+food_name 기준
        dedup = []
        seen_keys = set()
        for r in recipes:
            key = f"{r.get('source','')}|{r.get('food_name','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            dedup.append(r)

        aggregated = {"answer": answer, "recipes": dedup}
        return JSONResponse(content={"response": aggregated})


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

async def forward_to_text_service(message: str):
    """TextAgent Service로 텍스트 질의 전달"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"message": message}
            async with session.post(f"{TEXT_SERVICE_URL}/process", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"TextAgent Service 응답: {result}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"TextAgent Service 오류 (상태: {response.status}): {error_text}")
                    return {
                        "error": f"TextAgent Service 오류: {response.status}",
                        "message": error_text
                    }
    except aiohttp.ClientConnectorError as e:
        logger.error(f"TextAgent Service 연결 실패: {e}")
        return {
            "error": "TextAgent Service에 연결할 수 없습니다.",
            "message": "8002 서버가 실행 중인지 확인해주세요."
        }
    except Exception as e:
        logger.error(f"TextAgent Service 호출 중 오류: {e}")
        return {
            "error": "TextAgent Service 호출 중 오류가 발생했습니다.",
            "message": str(e)
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