import json
import logging
from typing import Dict

from .llm import LLMClient

logger = logging.getLogger(__name__)


class Substitutions:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def get_substitutions(self, dish: str, ingredient: str, user_substitute: str, message: str, context: str) -> Dict:
        target = ingredient or "핵심 재료"
        prompt = f"""
        당신은 프로 요리사입니다.
        대화 맥락: {context}
        사용자 원문: {message}
        요리: '{dish}'
        대체 대상 재료: '{target}'

        규칙:
        - 반드시 JSON만 출력하세요. JSON 이외의 텍스트/설명/추천/질문/코드블록 금지.
        - 셰프 이름, 도입부, 결론, 추가 제안 금지.
        - 사용자가 대체 재료를 명시했다면(substituteName가 비어있지 않다면) substitutes는 1개만 포함하고, 각 항목은 오직 method_adjustment(한 줄)만 포함하세요. 그 한 줄 안에 필요한 양/비율이 있다면 간단히 포함해도 됩니다.
        - 사용자가 대체 재료를 명시하지 않았다면 substitutes는 정확히 3개만 포함하고, 각 항목은 name, amount, method_adjustment 3개 필드만 포함하세요.

        출력 형식:
        {{
          "ingredient": "{target}",
          "substituteName": "{user_substitute}",
          "substitutes": [
            {{"name": "대체재1", "amount": "1:1", "method_adjustment": "조리법 조정"}}
          ]
        }}
        """
        data = self.llm.generate_json(prompt)
        if isinstance(data, dict):
            data.setdefault("ingredient", target)
            data.setdefault("substituteName", user_substitute or "")
            data.setdefault("substitutes", [])
            return data
        return {"ingredient": target, "substituteName": user_substitute or "", "substitutes": []}

    def get_necessity(self, dish: str, ingredient: str, context: str) -> Dict:
        target = ingredient or "핵심 재료"
        prompt = f"""
        당신은 프로 요리사입니다.
        대화 맥락: {context}
        요리: '{dish}'
        재료: '{target}'

        질문: '{target}'이(가) 반드시 필요한가?
        JSON으로만 출력하세요. 필드는 possible(불리언), flavor_change(문장 1줄)만 포함하세요. 다른 필드/설명은 금지.
        예시: {{"possible": true, "flavor_change": "감칠맛이 약간 줄어듭니다"}}
        """
        data = self.llm.generate_json(prompt)
        if isinstance(data, dict):
            return {
                "possible": bool(data.get("possible", False)),
                "flavor_change": str(data.get("flavor_change", "")),
            }
        return {"possible": False, "flavor_change": ""}


