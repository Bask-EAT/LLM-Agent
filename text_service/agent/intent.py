import logging
from typing import Literal

from .llm import LLMClient

logger = logging.getLogger(__name__)

Intent = Literal[
    "CATEGORY",
    "INGREDIENTS_TO_DISHES",
    "RECIPE",
    "INGREDIENTS",
    "TIP",
    "SUBSTITUTE",
    "NECESSITY",
    "OTHER",
]


class IntentClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def classify(self, message: str, context: str) -> Intent:
        prompt = f"""
        당신은 세계적으로 유명한 프로 셰프이자 요리 전문가입니다.
        Pierre Koffmann(프랑스), Gordon Ramsay(미국식), Ken Hom(중식), Massimo Bottura(이탈리아), José Andrés(스페인식), Yotam Ottolenghi(지중해식), 강레오(한식), 안성재(한식) 셰프의 경험과 스타일을 모두 갖춘 요리 컨설턴트입니다.

        대화 컨텍스트:
        {context}

        현재 메시지: {message}

        의도를 분류하세요:
        - CATEGORY: 음식 카테고리 요청 (한식 추천, 중식 추천, 프랑스식 추천, 이탈리아식 추천 등)
        - INGREDIENTS_TO_DISHES: 재료로 요리 추천 요청 (재료 가지고, 재료로 뭐 만들까, 재료로 할 수 있는 요리 등)
        - RECIPE: 레시피 요청 (레시피 알려줘, 조리법, 만드는 법, 그거 레시피, 그 음식 레시피, 레시피)
        - INGREDIENTS: 재료 요청 (재료 알려줘, 재료만, 그거 재료, 그 음식 재료, 재료)
        - TIP: 조리 팁 (팁 알려줘, 조리 팁, 그거 팁, 팁)
        - SUBSTITUTE: 재료 대체 요청 (X 대신, 대체 가능, 없으면 뭘로, 대체해도 돼, 대체 재료 등)
        - NECESSITY: 재료 필요 여부 요청 (꼭 넣어야 해?, 빼도 돼?, 없어도 돼?, 생략해도 돼?, 반드시 필요?)
        - OTHER: 그 외

        "그거", "그 음식", "이거" 같은 대명사는 이전 대화의 요리를 참조합니다.
        명시적인 지칭(요리명/카테고리/재료)이 없더라도 최근 대화 맥락을 활용해 의도를 분류하세요.
        카테고리를 언급하지 않고 "추천"만 요청해도 CATEGORY로 분류하세요(이 경우 추천은 기본적으로 한식 기준).
        출력은 CATEGORY, INGREDIENTS_TO_DISHES, RECIPE, INGREDIENTS, TIP, SUBSTITUTE, NECESSITY, OTHER 중 하나만
        """

        try:
            intent_text = self.llm.generate_text(prompt).upper()
            return intent_text if intent_text in {
                "CATEGORY",
                "INGREDIENTS_TO_DISHES",
                "RECIPE",
                "INGREDIENTS",
                "TIP",
                "SUBSTITUTE",
                "NECESSITY",
                "OTHER",
            } else "OTHER"  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "OTHER"  # type: ignore[return-value]


