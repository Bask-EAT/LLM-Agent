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
    예를 들어, 특정 요리의 레시피, 재료, 조리 팁을 물어보거나 음식 종류(한식, 중식 등)를 추천해달라고 할 때 유용합니다.
    유튜브 링크(URL)가 포함된 질문에는 이 도구를 사용하지 마세요.
    사용자의 질문을 그대로 입력값으로 사용하세요.
    """
    logger.info(
        f"TextAgent 도구 실행: '{query}'에 대한 처리를 위해 {TEXT_SERVICE_URL}/process로 전달합니다."
    )
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"message": query}
            logger.debug("=== 🤍payload for TextAgent Service: %s", payload)
            logger.info("=== 🤍TextAgent Service로 요청 전송: %s/process", TEXT_SERVICE_URL)
            async with session.post(f"{TEXT_SERVICE_URL}/process", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("TextAgent Service 응답: %s", result)
                    return json.dumps(result, ensure_ascii=False)
                else:
                    error_text = await response.text()
                    logger.error(
                        "TextAgent Service 오류 (상태: %s): %s", response.status, error_text
                    )
                    return json.dumps(
                        {"error": f"TextAgent Service 오류: {response.status}", "message": error_text},
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
            {"error": "TextAgent Service 호출 중 오류가 발생했습니다.", "message": str(e)},
            ensure_ascii=False,
        )


