# ingredient_service/tools.py
from langchain_core.tools import tool
import httpx
import logging
import os
from dotenv import load_dotenv
import json


# .env 파일의 환경 변수를 로드
load_dotenv()

# ingredient-service (8004번 포트)의 주소
INGREDIENT_SERVICE_URL = os.getenv("INGREDIENT_SERVICE_URL", "http://localhost:8004")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@tool
async def search_ingredient_by_text(query: str) -> dict:
    """사용자가 상품 정보를 찾거나, 장바구니에 상품을 담으려 할 때 사용합니다.

    ✅ 사용해야 하는 경우:
    - '계란 어디서 사야해', '소금 장바구니에 담아줘', '김치 가격 얼마야', '상품 주문하고 싶어'
    - '구매', '사야해', '어디서 사', '가격', '장바구니', '주문', '배송' 등의 키워드가 명확히 포함된 경우

    ❌ 절대 사용하지 마세요:
    - '김치전 레시피', '된장찌개 만드는 법', '요리법 알려줘', '재료는 뭐가 필요해' 등
    - 레시피, 조리법, 요리 관련 질문에는 절대 사용하지 마세요.

    이 도구는 오직 상품 구매/검색 목적으로만 사용해야 합니다."""

    logging.info(f"=== 🤍 [Agent Tool] search_ingredient_by_text 호출. 검색어: {query}")
    api_url = f"{INGREDIENT_SERVICE_URL}/search/text"
    payload = {"query": query}
    logging.info(
        f"=== 🤍 [Agent Tool] ingredient-service 서버로 요청 전송. URL: {api_url}, Payload: {payload}"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60.0)

            logging.info(
                f"=== 🤍 [Agent Tool] ingredient-service 서버로부터 응답 받음. Status Code: {response.status_code}"
            )
            response.raise_for_status()  # 오류가 있으면 예외 발생

            response_data = response.json()
            logging.info(
                f"=== 🤍 [Agent Tool] 에이전트에게 최종 결과 반환. 데이터: {response_data}"
            )

            return response_data

    except httpx.HTTPStatusError as e:
        error_message = f"ingredient-service 호출 중 HTTP 오류: {e.response.status_code} - {e.response.text}"
        logger.error(f"--- ❌ [Agent Tool] {error_message}")
        # 에이전트에게 오류를 텍스트로 반환하여, LLM이 문제를 인지하고 다른 행동을 하도록 유도할 수 있습니다.
        return {"error": error_message}
    except Exception as e:
        error_message = f"알 수 없는 오류 발생: {e}"
        logger.error(f"--- ❌ [Agent Tool] {error_message}", exc_info=True)
        return {"error": error_message}


@tool
async def search_ingredient_by_image(image_b64: str) -> dict:
    """사용자가 '이미지'만으로 재료나 상품 구매 정보를 물어볼 때 사용합니다. 이미지 자체에 대한 질문일 때 사용하세요."""

    try:
        if not image_b64:
            # 이 경우는 에이전트가 잘못 호출한 경우이므로, 에러를 명확히 반환합니다.
            return json.dumps({"error": "💢 호출 오류: 이미지 데이터가 비어있습니다. 💢"})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 모델이 base64 문자열 앞의 'data:image/jpeg;base64,' 부분을 포함해서 넘겨줄 수 있으므로, 순수 데이터만 추출합니다.
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]

            payload = {"image_b64": image_b64}
            logger.info(f"=== 💨 [Agent Tool] search_ingredient_by_image 호출. Payload: {payload}")
            response = await client.post(f"{INGREDIENT_SERVICE_URL}/search/image", json=payload)
            logger.info(f"=== 💨💨 [Agent Tool] Tool 에서 /search/image 응답 받음 {response}")
            response.raise_for_status()

            # 1. API 응답에서 파이썬 딕셔너리(내용물)를 추출합니다.
            result_data = response.json()

            # ⭐️ 핵심: 추출한 파이썬 딕셔너리를 표준 JSON 문자열로 변환하여 반환합니다.
            return json.dumps(result_data, ensure_ascii=False)
        
     # ⭐️ HTTP 요청 관련 예외를 여기서 직접 처리합니다!
    except httpx.HTTPStatusError as e:
        # 4xx, 5xx 에러가 발생하면, 안정적인 JSON 형식으로 에러 메시지를 반환합니다.
        error_content = e.response.json() if "application/json" in e.response.headers.get("content-type", "") else e.response.text
        logger.error(f"--- [Agent Tool] API 호출 실패 (HTTP {e.response.status_code}): {error_content}")
        return json.dumps({
            "error": f"이미지 분석 서비스에서 오류가 발생했습니다 (코드: {e.response.status_code}).",
            "detail": error_content
        })
    except Exception as e:
        # 그 외 네트워크 오류 등 모든 예외를 처리합니다.
        logger.error(f"--- [Agent Tool] 예상치 못한 오류 발생: {e}", exc_info=True)
        return json.dumps({"error": f"이미지 검색 중 알 수 없는 오류가 발생했습니다: {str(e)}"})


@tool
async def search_ingredient_multimodal(query: str, image_b64: str) -> dict:
    """사용자가 '텍스트와 이미지'를 모두 사용해 재료나 상품 구매 정보를 물어볼 때 사용합니다. 예를 들어 '이 사진 속 파스타면에 어울리는 소스 추천해줘' 같은 질문에 사용합니다."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/search/multimodal",
            params={"query": query, "image_b64": image_b64},
        )
        response.raise_for_status()
        return response.json()
