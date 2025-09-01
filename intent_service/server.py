from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from fastapi.responses import JSONResponse
from planning_agent import run_agent
import uuid
import time
import re
from typing import Optional

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



def count_youtube_urls(message: str) -> int:
    """메시지에 포함된 유튜브 URL의 개수를 반환합니다."""
    if not message:
        return 0
    # youtube.com/watch?v= 또는 youtu.be/ 패턴을 찾습니다.
    # re.findall은 모든 일치 항목을 리스트로 반환합니다.
    youtube_patterns = re.findall(r"(youtube\.com/watch\?v=|youtu\.be/)", message)
    return len(youtube_patterns)


# 백엔드 호환성을 위한 임시 결과 저장 (메모리 저장 최소화)
# 실제 운영에서는 데이터베이스에 저장해야 함
temp_results = {}

async def run_agent_direct(input_data: dict):
    """
    에이전트를 직접 실행하고 결과를 임시 저장 (백엔드 호환성)
    """
    logger.info(f"=== 🤍Direct-Agent: 작업 시작. ===")
    try:
        result = await run_agent(input_data)
        logger.info(f"=== 🤍❤ Agent 최종 응답: {result} ❤🤍 ===")
        logger.info(f"=== 🤍❤ Direct-Agent: 작업 완료. ❤🤍 ===")
        return result
    except Exception as e:
        logger.error(f"=== 🤍Direct-Agent: 작업 중 에러 발생: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}



# 쿼리 기반 처리: 메모리 저장 없이 즉시 응답
@app.post("/chat")
async def chat_with_agent(
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    사용자 요청(텍스트 및/또는 이미지)을 받아 즉시 처리하고 결과를 반환합니다.
    메모리에 상태를 저장하지 않고 쿼리 기반으로만 처리합니다.
    """
    try:
        logger.info(f"=== 🤍intent_service에서 /chat 엔드포인트 호출됨🤍 ===")
        logger.info(f"=== 🤍수신 메시지: {message}")
        if image:
            logger.info(f"=== 🤍수신 이미지: {image.filename}, {image.content_type}")
        if not message and not image:
            raise HTTPException(status_code=400, detail="message 또는 image가 필요합니다.")

        # 유튜브 URL 검사를 위해 메시지가 있으면 사용
        latest_message = message or ""

        # 유튜브 링크 개수 검사 로직 (실시간 입력만 체크)
        lines = latest_message.split('\n')
        real_time_message = lines[-1] if lines else ""
        
        if count_youtube_urls(real_time_message) > 1:
            logger.warning(f"요청 거부: 실시간 입력에 유튜브 링크가 2개 이상 포함됨 - {real_time_message}")
            raise HTTPException(
                status_code=400,
                detail="죄송합니다, 한 번에 하나의 유튜브 링크만 분석할 수 있습니다."
            )

        # 에이전트에 전달할 입력 데이터 구성
        input_data = {}
        if message:
            input_data["message"] = message
        if image:
            input_data["image"] = await image.read()
            input_data["image_filename"] = image.filename
            input_data["image_content_type"] = image.content_type

        # 직접 처리하고 결과를 임시 저장 (백엔드 호환성)
        result = await run_agent_direct(input_data)
        
        # 디버깅을 위한 로깅
        logger.info(f"=== 🤍Agent 처리 결과: {result}")
        logger.info(f"=== 🤍결과 타입: {type(result)}")
        
        # 백엔드 호환성을 위한 응답 형식 조정
        if isinstance(result, dict) and "error" in result:
            # 에러 응답
            logger.error(f"=== ❌ 에러 응답 반환: {result}")
            return JSONResponse(status_code=500, content=result)
        else:
            # job_id 생성 및 결과 임시 저장
            job_id = "direct_response_" + str(uuid.uuid4())
            temp_results[job_id] = {
                "status": "completed",
                "result": result
            }
            
            # 백엔드가 기대하는 202 + job_id 형식으로 응답
            formatted_result = {
                "job_id": job_id
            }
            logger.info(f"=== ✅ 백엔드 호환 응답 반환: {formatted_result}")
            return JSONResponse(status_code=202, content=formatted_result)
        
    except HTTPException as http_exc:
        logger.error(f"HTTP 예외 발생 (클라이언트로 전달됨): {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"예상치 못한 서버 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부에서 예상치 못한 오류가 발생했습니다: {e}")


# 백엔드 호환성을 위한 상태 조회 엔드포인트
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    백엔드 호환성을 위한 상태 조회 엔드포인트
    임시 저장된 결과를 반환
    """
    logger.info(f"=== 🤍상태 조회 요청: {job_id}")
    
    # 임시 저장된 결과 확인
    if job_id in temp_results:
        result = temp_results[job_id]
        logger.info(f"=== ✅ 저장된 결과 반환: {result}")
        
        # 결과 반환 후 임시 저장소에서 제거 (메모리 절약)
        del temp_results[job_id]
        
        return JSONResponse(content=result)
    else:
        # 알 수 없는 job_id
        logger.warning(f"=== ❌ 알 수 없는 job_id: {job_id}")
        return JSONResponse(status_code=404, content={"error": "Job not found"})


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "Intent LLM Server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 