import json
import logging
from typing import Dict, List

from .llm import LLMClient
from .parsers import (
    parse_recipe_from_text,
    parse_ingredients_from_text,
    parse_tips_from_text,
)

logger = logging.getLogger(__name__)


class Recipes:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def handle_vague_dish(self, dish: str, prefer_varieties: bool = False) -> Dict:
        prompt = f"""
        당신은 세계 각국의 요리법과 재료에 해박하며, 재료의 유무에 따른 대체재료(특히 한국에서 쉽게 구할 수 있는)까지 완벽하게 파악하고 있는 AI 셰프입니다. 복잡한 과정은 간단하게, 모든 이들이 쉽게 따라 할 수 있도록 명확하고 실용적인 레시피를 제공합니다.
        
        '{dish}'가 광범위한 요리 종류(예: 파스타, 볶음밥, 커리, 샐러드, 스테이크, 피자, 라면, 국수, 밥, 면, 탕, 찌개, 볶음, 구이 등)인지 판단하고:
        
        1. 만약 '{dish}'가 구체적인 요리명이라면, 완전한 레시피를 JSON으로 제공하세요.
        2. 만약 '{dish}'가 광범위한 요리 종류라면, 하위 요리 3~5가지를 JSON 배열로 출력하세요.
        3. "형용사/재료 + 요리명"으로 이루어진 복합 요리명(예: "소고기 김밥", "치즈 김밥", "봉골레 파스타", "버섯 리조또")은
           구체적인 단일 요리명으로 간주하고 1번 규칙을 따르세요(하위 종류를 묻지 말 것).

        추가 지시:
        - 사용자가 "추천"을 요청했거나, '{dish}'가 다양한 변형으로 자주 조합되는 요리(예: 김밥, 샌드위치, 파스타, 볶음밥 등)라면,
          구체 레시피 대신 하위 종류 3~5가지를 우선 출력하세요.
        
        **구체적인 요리명인 경우 레시피 JSON 형식:**
        {{
          "title": "{dish}",
          "ingredients": [
            {{"item": "재료명", "amount": "수량", "unit": "단위"}},
            {{"item": "재료명", "amount": "수량", "unit": "단위"}}
          ],
          "steps": ["1단계 설명", "2단계 설명", "3단계 설명"]
        }}
        
        **광범위한 요리 종류인 경우:**
        ["구체적인 요리명1", "구체적인 요리명2", "구체적인 요리명3"]
        
        **중요:** 반드시 유효한 JSON 형식으로만 응답하세요. JSON 이외의 텍스트나 설명은 절대 포함하지 마세요.
        """
        data = self.llm.generate_json(prompt)
        
        # 배열이 반환되면 모호한 요리로 처리
        if isinstance(data, list) and len(data) > 0:
            return {"title": f"{dish} 종류 추천", "varieties": data, "type": "vague_dish"}
        
        # 객체가 반환되면 구체적인 레시피로 처리
        if isinstance(data, dict) and data:
            data.setdefault("title", dish)
            # ingredients가 객체 배열이 아닌 경우 올바른 형식으로 변환
            if "ingredients" not in data or not isinstance(data["ingredients"], list) or not data["ingredients"]:
                data["ingredients"] = [{"item": "재료 정보를 찾을 수 없습니다", "amount": "", "unit": ""}]
            elif isinstance(data["ingredients"], list) and data["ingredients"] and isinstance(data["ingredients"][0], str):
                # 문자열 배열인 경우 객체 배열로 변환
                data["ingredients"] = [{"item": ing, "amount": "", "unit": ""} for ing in data["ingredients"]]
            
            data.setdefault("steps", ["조리법 정보를 찾을 수 없습니다"])
            if isinstance(data.get("steps"), list) and len(data["steps"]) > 15:
                data["steps"] = data["steps"][:15]
            return data
        
        # fallback
        return {
            "title": dish, 
            "ingredients": [{"item": "재료 정보를 찾을 수 없습니다", "amount": "", "unit": ""}], 
            "steps": ["조리법 정보를 찾을 수 없습니다"]
        }

    def get_recipe(self, dish: str) -> Dict:
        prompt = f"""
        당신은 세계 각국의 요리법과 재료에 해박하며, 재료의 유무에 따른 대체재료(특히 한국에서 쉽게 구할 수 있는)까지 완벽하게 파악하고 있는 AI 셰프입니다. 복잡한 과정은 간단하게, 모든 이들이 쉽게 따라 할 수 있도록 명확하고 실용적인 레시피를 제공합니다.

        '{dish}' 레시피를 JSON으로 작성하세요. '{dish}'가 요리명이 아닐 경우, 메시지에서 가장 가능성이 높은 단일 요리명 1개를 추론해 그 레시피만 작성하세요:
        - 조리법은 최대 15단계 이하로 작성
        - 복잡한 과정은 요약해서 핵심만 포함
        - 정확한 재료와 실용적인 조리법 제공
        - 레시피만 명확히 출력하며 팁/설명/도입문 등 불필요한 텍스트는 포함하지 마세요
        - 재료 형식 엄격 규칙:
          1) ingredients는 객체 배열이어야 합니다. 각 객체는 item/amount/unit 세 필드를 반드시 포함합니다.
          2) item: 재료명만 기재하세요. 손질/상태/형용사와 브랜드/원산지 정보는 제외합니다.
          3) amount: 수량 숫자만 기재하세요. 불명확하면 빈 문자열.
          4) unit: g, ml, 컵, 큰술, 작은술, 개, 마리, 통, 쪽, 톨 등. 단위가 없으면 빈 문자열.
          5) 범위 표기는 최소한의 숫자만 사용(예: 1~2개 → amount="1-2", unit="개").

        {{
          "title": "{dish}",
          "ingredients": {{"item": "재료명", "amount": "숫자만", "unit": "단위"}},
          "steps": ["1단계 설명", "2단계 설명"]
        }}
        """
        data = self.llm.generate_json(prompt)
        if isinstance(data, dict) and data:
            data.setdefault("title", dish)
            # ingredients가 객체 배열이 아닌 경우 올바른 형식으로 변환
            if "ingredients" not in data or not isinstance(data["ingredients"], list) or not data["ingredients"]:
                data["ingredients"] = [{"item": "재료 정보를 찾을 수 없습니다", "amount": "", "unit": ""}]
            elif isinstance(data["ingredients"], list) and data["ingredients"] and isinstance(data["ingredients"][0], str):
                # 문자열 배열인 경우 객체 배열로 변환
                data["ingredients"] = [{"item": ing, "amount": "", "unit": ""} for ing in data["ingredients"]]
            
            data.setdefault("steps", ["조리법 정보를 찾을 수 없습니다"])
            if isinstance(data.get("steps"), list) and len(data["steps"]) > 15:
                data["steps"] = data["steps"][:15]
            return data
        # fallback to parser
        # In case the model didn't return valid JSON
        # We cannot access the raw response here, so just return minimal structure
        return {
            "title": dish, 
            "ingredients": [{"item": "재료 정보를 찾을 수 없습니다", "amount": "", "unit": ""}], 
            "steps": ["조리법 정보를 찾을 수 없습니다"]
        }

    def get_ingredients(self, dish: str):
        prompt = f"""
        당신은 세계 각국의 요리법과 재료에 해박한 AI 셰프입니다.

        '{dish}'에 필요한 정확한 재료와 양을 JSON 객체 배열로만 출력하세요. '{dish}'가 애매하면 메시지에서 가장 가능성이 높은 단일 요리명 1개를 추론해 그 재료만 출력하세요.
        각 원소는 다음 형식의 객체여야 합니다: {"item": 재료명만, "amount": 숫자만, "unit": 단위만}
        - item: 수식어/브랜드/원산지/손질 상태를 제외한 재료명만
        - amount: 수량 숫자만(정수/소수/분수). 불명확하면 빈 문자열
        - unit: g, ml, 컵, 큰술, 작은술, 개, 마리, 통, 쪽, 톨 등 단위명. 없으면 빈 문자열
        기타 텍스트, 코드블록, 설명은 출력하지 마세요.
        예시: [{"item": "재료1", "amount": "100", "unit": "g"}, {"item": "재료2", "amount": "1/2", "unit": "컵"}]
        """
        data = self.llm.generate_json(prompt)
        if isinstance(data, list) and len(data) > 0:
            return data
        return ["재료 정보를 찾을 수 없습니다"]

    def get_tips(self, dish: str) -> List[str]:
        prompt = f"""
        당신은 세계 각국의 요리법에 해박한 AI 셰프입니다.

        '{dish}'를 더 맛있게 만드는 실용적인 조리 팁 3개를 JSON 배열로 출력하세요.
        각 팁은 구체적이고 실용적이어야 합니다.
        예시: ["구체적인 팁1", "실용적인 팁2", "전문가 팁3"]
        """
        data = self.llm.generate_json(prompt)
        if isinstance(data, list) and len(data) > 0:
            return data
        return ["조리 팁을 찾을 수 없습니다"]


