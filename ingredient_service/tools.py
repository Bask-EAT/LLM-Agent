# ingredient_service/tools.py
from langchain_core.tools import tool
import httpx
import logging
import os

# --- 설정 (파일 상단에 위치) ---
# ingredient-service (8004번 포트)의 주소
# .env 파일에 INGREDIENT_SERVICE_URL="http://localhost:8004" 와 같이 설정하는 것을 권장합니다.
INGREDIENT_SERVICE_URL = os.getenv("INGREDIENT_SERVICE_URL", "http://localhost:8004")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@tool
async def search_ingredient_by_text(query: str) -> dict:
    """사용자가 상품 정보를 찾거나, 장바구니에 상품을 담으려 할 때 사용합니다. '계란 찾아줘', '소금 장바구니에 담아줘' 와 같은 요청을 처리합니다."""
    
    logging.info(f"=== 🤍search_ingredient_by_text 호출. 검색어: {query}")
    api_url = f"{INGREDIENT_SERVICE_URL}/search/text"
    payload = {"query": query} # API 엔드포인트에 맞게 payload 수정
    logging.info(f"=== 🤍 [Tool Request] ingredient-service 서버로 요청 전송. URL: {api_url}, Payload: {payload}")

    try:
        # 2. httpx를 사용해 ingredient-service API를 호출합니다.
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60.0)
            
            logging.info(f"=== 🤍 [Tool Response] ingredient-service 서버로부터 응답 받음. Status Code: {response.status_code}")
            response.raise_for_status() # 오류가 있으면 예외 발생
            
            response_data = response.json()
            logging.info(f"=== 🤍 [Tool Return] 에이전트에게 최종 결과 반환. 데이터: {response_data}")
            
            return response_data
        
    except httpx.HTTPStatusError as e:
        error_message = f"ingredient-service 호출 중 HTTP 오류: {e.response.status_code} - {e.response.text}"
        logger.error(f"--- ❌ [Tool Error] {error_message}")
        # 에이전트에게 오류를 텍스트로 반환하여, LLM이 문제를 인지하고 다른 행동을 하도록 유도할 수 있습니다.
        return {"error": error_message}
    except Exception as e:
        error_message = f"알 수 없는 오류 발생: {e}"
        logger.error(f"--- ❌ [Tool Error] {error_message}", exc_info=True)
        return {"error": error_message}


@tool
async def search_ingredient_by_image(image_b64: str) -> dict:
    """사용자가 '이미지'만으로 재료나 상품 구매 정보를 물어볼 때 사용합니다. 이미지 자체에 대한 질문일 때 사용하세요."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/search/image", params={"image_b64": image_b64})
        response.raise_for_status()
        return response.json()

@tool
async def search_ingredient_multimodal(query: str, image_b64: str) -> dict:
    """사용자가 '텍스트와 이미지'를 모두 사용해 재료나 상품 구매 정보를 물어볼 때 사용합니다. 예를 들어 '이 사진 속 파스타면에 어울리는 소스 추천해줘' 같은 질문에 사용합니다."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/search/multimodal", params={"query": query, "image_b64": image_b64})
        response.raise_for_status()
        return response.json()