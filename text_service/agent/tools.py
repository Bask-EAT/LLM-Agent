import json
import logging
import aiohttp
from langchain_core.tools import tool

from .constants import TEXT_SERVICE_URL

logger = logging.getLogger(__name__)


@tool
async def text_based_cooking_assistant(query: str) -> str:
    """
    텍스트 기반의 요리 관련 질문에 답변할 때 사용합니다.

    ✅ 사용해야 하는 경우 (우선순위 높음):
    - '김치전 레시피', '된장찌개 만드는 법', '요리법 알려줘', '재료는 뭐가 필요해'
    - '레시피', '만드는 법', '조리법', '재료', '요리', '만들어', '어떻게' 등의 키워드가 포함된 모든 질문
    - 특정 요리의 레시피, 재료, 조리 팁을 물어보거나 음식 종류(한식, 중식 등)를 추천해달라고 할 때

    ❌ 사용하지 마세요:
    - 유튜브 링크(URL)가 포함된 질문
    - 상품 구매/검색 관련 질문 ('어디서 사야해', '가격 얼마야', '장바구니에 담아줘' 등)

    이 도구는 모든 요리/레시피 관련 질문의 기본 도구입니다.
    사용자의 질문을 그대로 입력값으로 사용하세요.
    """
    logger.info(
        f"TextAgent 도구 실행: '{query}'에 대한 처리를 위해 {TEXT_SERVICE_URL}/process로 전달합니다."
    )
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"message": query}
            logger.debug("=== 🤍payload for TextAgent Service: %s", payload)
            logger.info(
                "=== 🤍TextAgent Service로 요청 전송: %s/process", TEXT_SERVICE_URL
            )
            async with session.post(
                f"{TEXT_SERVICE_URL}/process", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("TextAgent Service 응답: %s", result)
                    return json.dumps(result, ensure_ascii=False)
                else:
                    error_text = await response.text()
                    logger.error(
                        "TextAgent Service 오류 (상태: %s): %s",
                        response.status,
                        error_text,
                    )
                    return json.dumps(
                        {
                            "error": f"TextAgent Service 오류: {response.status}",
                            "message": error_text,
                        },
                        ensure_ascii=False,
                    )
    except aiohttp.ClientConnectorError as e:
        logger.error(f"TextAgent Service 연결 실패: {e}")
        return json.dumps(
            {
                "error": "TextAgent Service에 연결할 수 없습니다.",
                "message": "8002 서버가 실행 중인지 확인해주세요.",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"TextAgent Service 호출 중 오류: {e}")
        return json.dumps(
            {
                "error": "TextAgent Service 호출 중 오류가 발생했습니다.",
                "message": str(e),
            },
            ensure_ascii=False,
        )
