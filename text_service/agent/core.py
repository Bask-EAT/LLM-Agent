import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from ..config import GEMINI_API_KEY
import google.generativeai as genai

from .constants import (
    CACHE_TTL_SECONDS,
    CUISINE_PROFILES,
    EXPLICIT_NEW_INTENT_KEYWORDS,
    NON_STYLE_HINTS,
    OTHER_REQUEST_KEYWORDS,
    STYLE_KEYWORDS,
)
from .llm import LLMClient
from .intent import IntentClassifier
from .recommenders import Recommenders
from .recipes import Recipes
from .substitutions import Substitutions
from .extractors import (
    find_dish_by_pattern,
    match_ingredient_from_inventory,
    map_to_inventory,
    PRONOUNS,
)


logger = logging.getLogger(__name__)


class TextAgent:
    def __init__(self) -> None:
        genai.configure(api_key=GEMINI_API_KEY)
        self.llm = LLMClient()
        self.intent_classifier = IntentClassifier(self.llm)
        self.recommenders = Recommenders(self.llm)
        self.recipes = Recipes(self.llm)
        self.substitutions = Substitutions(self.llm)
        
        # 메모리 저장 비활성화: 모든 상태는 쿼리 기반으로만 처리
        # 데이터베이스에만 저장, 메모리에는 상태 저장하지 않음



    def _has_explicit_new_intent(self, message: str) -> bool:
        """명시적 새 의도 키워드 감지"""
        text = (message or "").lower()
        return any(k in text for k in EXPLICIT_NEW_INTENT_KEYWORDS)

    def _is_other_request(self, message: str) -> bool:
        """사용자가 '다른 거' 계열을 요청했는지 여부"""
        text = (message or "").lower().strip()
        if not text:
            return False
        return any(k in text for k in OTHER_REQUEST_KEYWORDS)

    def _is_style_followup(self, message: str) -> bool:
        """스타일 후속 요청인지 확인"""
        text = (message or "").lower().strip()
        if not text:
            return False
        has_style = any(k in text for k in STYLE_KEYWORDS)
        has_non_style = any(k in text for k in NON_STYLE_HINTS)
        return has_style and not has_non_style

    async def process_message(self, message: str) -> Dict[str, Any]:
        """쿼리 기반 처리: 메모리 상태에 의존하지 않고 입력 메시지만으로 처리"""
        try:
            logger.info(f"🔍 [TextAgent] 쿼리 기반 메시지 처리 시작: '{message}'")

            # 메시지에서 직접 정보 추출하여 처리
            intent = self.intent_classifier.classify(message, "")
            logger.info(f"🔍 [TextAgent] 분류된 의도: {intent}")

            if intent == "CATEGORY":
                # 메시지에서 직접 스타일 정보 추출
                if self._is_style_followup(message):
                    # 스타일 후속 요청인 경우 메시지에서 재료 정보 추출
                    extracted_ingredients = self._extract_ingredients_from_message(message)
                    if extracted_ingredients:
                        result = await self.recommend_dishes_by_ingredients_with_style(message, extracted_ingredients)
                        response_text = result.get("answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다.")
                        return {"answer": response_text, "food_name": None, "ingredients": result.get("extracted_ingredients", extracted_ingredients), "recipe": []}
                
                # 일반 카테고리 추천
                result = self.recommenders.recommend_by_category(message, avoid=[])
                items = result.get("items", []) if isinstance(result, dict) else []
                if not isinstance(result, dict) or not items or (result.get("category") == "미정"):
                    response_text = "혹시 특별히 끌리는 요리 스타일(한식, 중식, 이탈리아식 등)이 있으신가요? 말씀해주시면 거기에 맞춰 맛있는 메뉴를 추천해드릴게요!"
                else:
                    category_label = result.get("category", "한식")
                    response_text = ""
                    if category_label == "한식":
                        for i, item in enumerate(items, 1):
                            name = item if isinstance(item, str) else item.get("name", "")
                            if name:
                                response_text += f"{i}. {name}\n"
                    else:
                        for i, item in enumerate(items, 1):
                            if isinstance(item, dict):
                                name = item.get("name", "")
                                desc = item.get("description", "")
                                line = name
                                if desc:
                                    line += f" — {desc}"
                                if line.strip():
                                    response_text += f"{i}. {line}\n"
                            else:
                                response_text += f"{i}. {item}\n"
                    if not response_text.strip():
                        response_text = "죄송합니다. 추천 요리를 찾을 수 없습니다."
                    else:
                        response_text += "\n원하는 요리의 레시피를 알려드릴까요? 번호나 요리명을 말씀해 주세요."

                return {"answer": response_text, "food_name": None, "ingredients": [], "recipe": []}

            elif intent == "INGREDIENTS_TO_DISHES":
                result = await self.recommend_dishes_by_ingredients(message)
                response_text = result.get("answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다.")
                extracted_ingredients = result.get("extracted_ingredients", [])
                return {"answer": response_text, "food_name": None, "ingredients": [], "recipe": []}

            elif intent == "RECIPE":
                dish = self._extract_dish_smart(message)
                result = self.recipes.handle_vague_dish(dish)
                if result.get("type") == "vague_dish":
                    varieties = result.get("varieties", [])
                    response_text = f"어떤 {dish} 레시피를 원하시나요?\n\n"
                    for i, variety in enumerate(varieties, 1):
                        response_text += f"{i}. {variety}\n"
                    response_text += f"\n다른 원하시는 {dish} 종류가 있으시면 말씀해주세요!"
                    return {"answer": response_text, "food_name": None, "ingredients": [], "recipe": []}
                else:
                    title = result.get("title", dish)
                    ingredients = result.get("ingredients", [])
                    steps = result.get("steps", [])
                    response_text = "📋 [재료]\n"
                    for i, ingredient in enumerate(ingredients, 1):
                        response_text += f"{i}. {ingredient}\n"
                    response_text += "\n👨‍🍳 [조리법]\n"
                    for i, step in enumerate(steps, 1):
                        response_text += f"{i}. {step}\n"
                    simple_answer = f"네. {title}의 레시피를 알려드릴게요."
                    return {"answer": simple_answer, "food_name": title, "ingredients": ingredients, "recipe": steps}

            elif intent == "INGREDIENTS":
                dish = self._extract_dish_smart(message)
                result = self.recipes.get_ingredients(dish)
                if not result or result == ["재료 정보를 찾을 수 없습니다"]:
                    response_text = "재료 정보를 찾을 수 없습니다"
                else:
                    response_lines = []
                    for i, ingredient in enumerate(result, 1):
                        response_lines.append(f"{i}. {ingredient}")
                    response_text = "\n".join(response_lines)
                return {"answer": response_text, "food_name": dish, "ingredients": result, "recipe": []}

            elif intent == "TIP":
                dish = self._extract_dish_smart(message)
                result = self.recipes.get_tips(dish)
                if not result or result == ["조리 팁을 찾을 수 없습니다"]:
                    response_text = f"죄송합니다. {dish}의 조리 팁을 찾을 수 없습니다."
                else:
                    response_text = f"네, 알겠습니다! {dish}를 더 맛있게 만드는 조리 팁입니다.\n\n"
                    response_text += "💡 [조리 팁]\n"
                    for i, tip in enumerate(result, 1):
                        response_text += f"{i}. {tip}\n"
                    response_text += f"\n{dish} 레시피나 재료도 궁금하시면 말씀해주세요!"
                return {"answer": response_text, "food_name": dish, "ingredients": [], "recipe": result}

            elif intent == "SUBSTITUTE":
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                user_substitute = self._extract_explicit_substitute_name(message)
                subs = self.substitutions.get_substitutions(dish, ingredient, user_substitute, message, "")
                target_ing = subs.get("ingredient", ingredient or "해당 재료")
                substitute_name = subs.get("substituteName", user_substitute or "")
                candidates = subs.get("substitutes", [])
                if not candidates:
                    response_text = ""
                else:
                    if substitute_name:
                        method = (candidates[0].get("method_adjustment", "") or "").strip()
                        response_text = method
                    else:
                        lines = []
                        for i, item in enumerate(candidates, 1):
                            name = item.get("name", "")
                            amount = item.get("amount", item.get("ratio_or_amount", ""))
                            method = item.get("method_adjustment", item.get("method", ""))
                            parts = [str(i) + ".", name]
                            if amount:
                                parts.append(amount)
                            if method:
                                parts.append(method)
                            line = " — ".join([p for p in parts if p])
                            if line.strip():
                                lines.append(line)
                        response_text = "\n".join(lines)
                return {"answer": response_text, "food_name": dish, "ingredients": [target_ing], "recipe": []}

            elif intent == "NECESSITY":
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                result = self.substitutions.get_necessity(dish, ingredient, "")
                possible = result.get("possible", False)
                flavor_change = result.get("flavor_change", "")
                response_text = f"가능: {'예' if possible else '아니오'}"
                if flavor_change:
                    response_text += f"\n맛 변화: {flavor_change}"
                return {"answer": response_text, "food_name": dish, "ingredients": [ingredient] if ingredient else [], "recipe": []}

            else:
                response_text = "요리 관련 질문을 해주세요. 레시피, 재료, 조리 팁 등 무엇이든 도와드릴 수 있어요!"
                return {"answer": response_text, "food_name": None, "ingredients": [], "recipe": []}
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            error_message = "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            return {"answer": error_message, "food_name": None, "ingredients": [], "recipe": []}

    def _extract_dish_smart(self, message: str) -> str:
        """메시지에서 직접 요리명 추출 (메모리 상태에 의존하지 않음)"""
        dish = find_dish_by_pattern(message)
        if dish:
            return dish
        return "알 수 없는 요리"
    
    def _extract_ingredients_from_message(self, message: str) -> List[str]:
        """메시지에서 직접 재료 정보 추출"""
        # 간단한 재료 추출 로직 (실제 구현에서는 더 정교한 NLP 사용 가능)
        import re
        ingredients = []
        
        # 일반적인 재료 패턴 매칭
        ingredient_patterns = [
            r'([가-힣]+)\s*(\d+[가-힣]*)\s*(개|마리|장|줄기|뿌리|개)',
            r'([가-힣]+)\s*(\d+[가-힣]*)\s*(g|kg|ml|l|컵|큰술|작은술)',
            r'([가-힣]+)\s*(약간|조금|적당히)',
            r'([가-힣]+)\s*(\d+[가-힣]*)',
        ]
        
        for pattern in ingredient_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if len(match) >= 2:
                    ingredient = f"{match[0]} {match[1]}"
                    if len(match) > 2:
                        ingredient += f" {match[2]}"
                    ingredients.append(ingredient.strip())
        
        return ingredients

    def _extract_ingredient_to_substitute(self, message: str) -> str:
        """메시지에서 대체할 재료 추출 (메모리 상태에 의존하지 않음)"""
        patterns = [
            r"([가-힣A-Za-z]+)\s*대신",
            r"([가-힣A-Za-z]+)\s*없으면",
            r"([가-힣A-Za-z]+)\s*대체",
            r"([가-힣A-Za-z]+)\s*못\s*먹",
            r"([가-힣A-Za-z]+)\s*빼고",
            r"([가-힣A-Za-z]+)\s*말고",
            r"([가-힣A-Za-z]+)\s*빼도\s*돼",
            r"([가-힣A-Za-z]+)\s*생략\s*해도\s*돼",
        ]
        import re
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                raw_ing = match.group(1).strip()
                if 1 < len(raw_ing) < 30:
                    return raw_ing
        return ""

    def _extract_explicit_substitute_name(self, message: str) -> str:
        import re
        patterns = [
            r"[가-힣A-Za-z]+\s*(?:말고|대신|빼고)\s*([가-힣A-Za-z]+)",
            r"\?\s*([가-힣A-Za-z]+)\s*써도\s*돼",
        ]
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                name = match.group(1).strip()
                if 1 < len(name) < 30:
                    return name
        return ""

    async def recommend_dishes_by_ingredients(self, message: str) -> Dict:
        """재료 기반 요리 추천 (메모리 상태에 의존하지 않음)"""
        prompt = f"""
        당신은 한식 전문가입니다. 사용자가 요청한 재료로 만들 수 있는 한식 요리를 추천해주세요.

        사용자 메시지: "{message}"

        **중요한 규칙:**
        1. 반드시 유효한 JSON 형식으로만 응답하세요
        2. JSON 이외의 텍스트나 설명은 절대 포함하지 마세요
        3. 코드 블록(```)이나 다른 마크다운 문법을 사용하지 마세요
        4. 재료가 명확하지 않으면 빈 배열로 설정하세요

        **작업 순서:**
        1. 메시지에서 언급된 재료를 추출하세요
        2. 해당 재료를 주재료로 사용하는 한식 요리 3가지를 추천하세요
        3. 각 요리 별로 간단한 소개를 하세요

        **JSON 응답 형식 (이 형식을 정확히 따르세요):**
        {{
          "ingredients": ["재료1", "재료2"],
          "dishes": [
            {{
              "name": "요리명1",
              "description": "한 줄 소개"
            }},
            {{
              "name": "요리명2", 
              "description": "한 줄 소개"
            }},
            {{
              "name": "요리명3", 
              "description": "한 줄 소개"
            }}
          ]
        }}
        """
        data = self.llm.generate_json(prompt)
        if not isinstance(data, dict):
            return {"answer": "재료 분석 중 오류가 발생했습니다. 다시 시도해주세요.", "extracted_ingredients": []}
        ingredients = data.get("ingredients", [])
        dishes = data.get("dishes", [])
        
        if not dishes:
            return {"answer": "해당 재료로 만들 수 있는 요리를 찾을 수 없습니다.", "extracted_ingredients": ingredients, "food_name": None, "recipe": []}
        
        response_text = f"다음 재료들로 만들 수 있는 한식 요리를 추천드려요:\n\n"
        response_text += "\n🍳 [추천 요리]\n"
        for i, dish in enumerate(dishes, 1):
            if isinstance(dish, dict):
                name = (dish.get("name") or "").strip()
                desc = (dish.get("description") or dish.get("note") or dish.get("uses") or "").strip()
                line = f"{i}. {name}" if name else f"{i}."
                if desc:
                    line += f" — {desc}"
                response_text += line + "\n"
            else:
                response_text += f"{i}. {dish}\n"
        response_text += "\n원하는 요리 형식이 있으신가요? (프랑스식, 이탈리아식, 미국식 등)"
        response_text += "\n또는 위 요리 중 어떤 것의 레시피를 알고 싶으시면 번호나 요리명을 말씀해주세요!"
        
        return {"answer": response_text, "extracted_ingredients": ingredients, "recommended_dishes": dishes}

    async def recommend_dishes_by_ingredients_with_style(self, message: str, ingredients: List[str]) -> Dict:
        """스타일 기반 재료 요리 추천 (메모리 상태에 의존하지 않음)"""
        lower_msg = message.lower()
        inferred = next((c for c in CUISINE_PROFILES if any(k in lower_msg or k in message for k in c["keywords"])), None)
        if inferred is None:
            return {"answer": "원하시는 요리 스타일을 알려주세요. (예: 프랑스식, 이탈리아식, 미국식 등)", "extracted_ingredients": ingredients}
        
        category_key = inferred["key"]
        ingredients_str = ", ".join(ingredients)
        prompt = f"""
        당신은 {category_key} 요리 전문가입니다. 기존 재료를 활용해 {category_key} 스타일 요리를 추천해주세요.

        기존 재료: {ingredients_str}
        요청된 스타일: {category_key}
        사용자 메시지: "{message}"

        중요한 규칙:
        - 반드시 유효한 JSON 형식으로만 응답하세요
        - JSON 이외의 텍스트나 설명은 절대 포함하지 마세요
        - 코드 블록이나 다른 마크다운 문법을 사용하지 마세요
        - 위 재료들을 반드시 주재료로 사용하는 {category_key} 요리만 추천하세요
        - 모든 출력은 한국어로 작성하세요. 요리명은 한국어 표기를 우선 사용하고, 필요하면 괄호에 원어를 병기하세요.

        JSON 응답 형식(정확히 따르세요):
        {{
          "style": "{category_key}",
          "dishes": [
            {{"name": "요리명1", "description": "한 줄 소개"}},
            {{"name": "요리명2", "description": "한 줄 소개"}},
            {{"name": "요리명3", "description": "한 줄 소개"}}
          ]
        }}
        """
        data = self.llm.generate_json(prompt)
        if not isinstance(data, dict):
            return {"answer": f"{category_key} 스타일 요리 추천 중 오류가 발생했습니다. 다시 시도해주세요.", "extracted_ingredients": ingredients}
        
        style = data.get("style", category_key)
        dishes = data.get("dishes", [])
        
        if not dishes:
            return {"answer": f"해당 재료로 만들 수 있는 {category_key} 요리를 찾을 수 없습니다.", "extracted_ingredients": ingredients}
        
        response_text = f"{style} 스타일 추천 요리:\n\n"
        for i, dish in enumerate(dishes, 1):
            if isinstance(dish, dict):
                name = (dish.get("name") or "").strip()
                desc = (dish.get("description") or dish.get("note") or dish.get("uses") or "").strip()
                line = f"{i}. {name}" if name else f"{i}."
                if desc:
                    line += f" — {desc}"
                response_text += line + "\n"
            else:
                response_text += f"{i}. {dish}\n"
        response_text += "\n원하는 요리의 레시피를 알려드릴까요? 번호(예: 1번)나 요리명을 말씀해 주세요."
        
        return {"answer": response_text, "extracted_ingredients": ingredients, "style": style, "recommended_dishes": dishes}

    async def _handle_selection_if_any(self, message: str):
        """번호 선택 처리 (메모리 상태에 의존하지 않음)"""
        import re
        text = (message or "").strip()
        logger.info(f"🔍 [TextAgent] 번호 선택 체크 시작: '{text}'")
        
        if not text or not re.search(r"\d", text):
            return None
            
        # 메시지에서 숫자 추출
        indices = re.findall(r"\d+", text)
        if not indices:
            return None
            
        # 숫자가 포함된 경우 일반적인 요리명으로 처리
        # 예: "1번" -> "1번 요리" 또는 "첫 번째 요리"
        number = indices[0]
        try:
            num = int(number)
            if 1 <= num <= 10:  # 1-10번까지만 처리
                # 번호를 요리명으로 변환하는 간단한 로직
                dish_names = ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", 
                            "여섯 번째", "일곱 번째", "여덟 번째", "아홉 번째", "열 번째"]
                if num <= len(dish_names):
                    dish_name = f"{dish_names[num-1]} 요리"
                    recipe = self.recipes.get_recipe(dish_name)
                    if recipe and recipe.get("title"):
                        return {"answer": f"네. {recipe.get('title')}의 레시피를 알려드릴게요.", 
                               "food_name": recipe.get("title"), 
                               "ingredients": recipe.get("ingredients", []), 
                               "recipe": recipe.get("steps", [])}
        except ValueError:
            pass
            
        return None


