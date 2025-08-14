# ingredient_service/tools.py
from langchain_core.tools import tool
import httpx
import logging

# ingredient_service가 실행되는 API 서버 주소
# 로컬에서 테스트 시 uvicorn으로 실행한 주소를 적어주세요
API_BASE_URL = "http://127.0.0.1:8004" # <- 이 주소는 실제 환경에 맞게 수정해야 합니다.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
async def search_ingredient_by_text(query: str) -> dict:
    """사용자가 상품 정보를 찾거나, 장바구니에 상품을 담으려 할 때 사용합니다. '계란 찾아줘', '소금 장바구니에 담아줘' 와 같은 요청을 처리합니다."""
    
    logging.info(f"=== 🧡search_ingredient_by_text 호출. 검색어: {query}")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/search/text", params={"query": query})
        response.raise_for_status() # 오류가 있으면 예외 발생
        return response.json()

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