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

        self.conversation_history: List[Dict[str, str]] = []
        self.last_dish: Optional[str] = None
        self.last_ingredients: List[str] = []
        self.last_intent: Optional[str] = None
        self.last_suggested_dishes: List[str] = []
        self.last_ingredients_ts: float = 0.0
        self.last_suggested_ts: float = 0.0
        self.last_suggested_turn: int = 0
        self.cache_ttl_sec: int = CACHE_TTL_SECONDS
        self.turn_idx: int = 0
        self.last_ingredients_turn: int = 0
        self.last_style: str = ""
        self.last_style_ts: float = 0.0

    def _add_assistant_response(self, content: str):
        # 히스토리 비활성화: 서버는 대화 로그를 저장하지 않음
        return

    def _get_recent_context(self, count: int = 3) -> str:
        # 히스토리 비활성화: 항상 빈 컨텍스트 반환
        return ""

    def _is_fresh(self, ts: float) -> bool:
        if not ts:
            return False
        return (time.time() - ts) <= self.cache_ttl_sec

    def _has_explicit_new_intent(self, message: str) -> bool:
        text = (message or "").lower()
        return any(k in text for k in EXPLICIT_NEW_INTENT_KEYWORDS)

    def _is_other_in_same_style(self, message: str) -> bool:
        text = (message or "").lower().strip()
        if not text:
            return False
        return (
            any(k in text for k in OTHER_REQUEST_KEYWORDS)
            and bool(self.last_style)
            and self._is_fresh(self.last_style_ts)
        )

    def _is_other_request(self, message: str) -> bool:
        """사용자가 '다른 거' 계열을 요청했는지 여부(스타일 보유 여부 무관)."""
        text = (message or "").lower().strip()
        if not text:
            return False
        return any(k in text for k in OTHER_REQUEST_KEYWORDS)

    def _is_cache_valid(self) -> bool:
        time_ok = self._is_fresh(self.last_ingredients_ts)
        turn_ok = (
            (self.turn_idx - self.last_ingredients_turn) <= 3
            if self.last_ingredients_turn
            else False
        )
        return time_ok and turn_ok and bool(self.last_ingredients)

    def _is_style_followup(self, message: str) -> bool:
        text = (message or "").lower().strip()
        if not text:
            return False
        has_style = any(k in text for k in STYLE_KEYWORDS)
        has_non_style = any(k in text for k in NON_STYLE_HINTS)
        recent_allows_follow = self.last_intent in {
            "INGREDIENTS_TO_DISHES",
            "RECIPE",
            "INGREDIENTS",
        }
        return (
            has_style
            and not has_non_style
            and recent_allows_follow
            and bool(self.last_ingredients)
            and self._is_fresh(self.last_ingredients_ts)
        )

    async def process_message(self, message: str) -> Dict[str, Any]:
        try:
            self.turn_idx += 1
            # 히스토리 비활성화: 사용자 메시지 저장하지 않음

            selection_result = await self._handle_selection_if_any(message)
            if selection_result is not None:
                return selection_result

            if self._has_explicit_new_intent(message):
                self.last_ingredients = []
                self.last_ingredients_ts = 0.0
                self.last_ingredients_turn = 0
                self.last_suggested_dishes = []
                self.last_suggested_ts = 0.0

            if self._is_other_in_same_style(message) and self._is_cache_valid():
                result = await self.recommend_dishes_by_ingredients_with_style(
                    message, self.last_ingredients
                )
                response_text = result.get("answer", "추천을 찾을 수 없습니다.")
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": result.get(
                        "extracted_ingredients", self.last_ingredients
                    ),
                    "recipe": [],
                }

            # last_style이 없더라도 '다른 거'라면, 최근 재료 캐시로 동일 맥락 재추천(스타일 가정 없음)
            if (
                self._is_other_request(message)
                and self._is_cache_valid()
                and not self.last_style
            ):
                ingredients_str = ", ".join(
                    [
                        (
                            ing.get("item")
                            if isinstance(ing, dict) and ing.get("item")
                            else str(ing)
                        )
                        for ing in self.last_ingredients
                    ]
                )
                result = await self.recommend_dishes_by_ingredients(ingredients_str)
                response_text = result.get("answer", "추천을 찾을 수 없습니다.")
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": result.get(
                        "extracted_ingredients", self.last_ingredients
                    ),
                    "recipe": [],
                }

            if self._is_style_followup(message) and self._is_cache_valid():
                result = await self.recommend_dishes_by_ingredients_with_style(
                    message, self.last_ingredients
                )
                response_text = result.get(
                    "answer", "해당 재료로 만들 수 있는 요리를 찾을 수 없습니다."
                )
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": result.get(
                        "extracted_ingredients", self.last_ingredients
                    ),
                    "recipe": [],
                }

            # 동일 스타일 내 '다른 거' 요청이지만 재료 캐시가 없을 때: 최근 스타일로 카테고리 재추천
            if (
                self._is_other_request(message)
                and getattr(self, "last_style", "")
                and not self._is_cache_valid()
            ):
                synthetic_message = self.last_style
                result = self.recommenders.recommend_by_category(
                    synthetic_message, avoid=self.last_suggested_dishes
                )
                items = result.get("items", []) if isinstance(result, dict) else []
                category_label = (
                    result.get("category", self.last_style)
                    if isinstance(result, dict)
                    else self.last_style
                )

                # 디듀프: 이전 제안 및 현재 목록 내 중복 제거
                def _norm_cat(n: str) -> str:
                    import re

                    s = (n or "").lower().strip()
                    s = re.sub(r"[\s·ㆍ・/|()-]+", "", s)
                    s = re.sub(r"[^a-z가-힣]", "", s)
                    return s

                avoid_set = {
                    _norm_cat(x) for x in getattr(self, "last_suggested_dishes", [])
                }
                seen = set()
                deduped = []
                for it in items:
                    name = (
                        it
                        if isinstance(it, str)
                        else (it.get("name", "") if isinstance(it, dict) else str(it))
                    )
                    key = _norm_cat(name)
                    if not name or key in seen or key in avoid_set:
                        continue
                    seen.add(key)
                    deduped.append(it)
                items = deduped
                response_text = ""
                if category_label == "한식":
                    for i, item in enumerate(items, 1):
                        name = (
                            item
                            if isinstance(item, str)
                            else (
                                item.get("name", "")
                                if isinstance(item, dict)
                                else str(item)
                            )
                        )
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

                self._add_assistant_response(response_text)
                self.last_intent = "CATEGORY"
                self.last_style = category_label or self.last_style
                self.last_style_ts = time.time()
                if category_label == "한식":
                    self.last_suggested_dishes = [
                        str(x).strip()
                        for x in items
                        if (isinstance(x, str) and x.strip())
                        or (isinstance(x, dict) and x.get("name"))
                    ]
                else:
                    self.last_suggested_dishes = [
                        x.get("name", "").strip()
                        for x in items
                        if isinstance(x, dict) and x.get("name")
                    ]
                self.last_suggested_ts = time.time()
                self.last_suggested_turn = self.turn_idx
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": [],
                    "recipe": [],
                }

            intent = self.intent_classifier.classify(message, "")

            if intent == "CATEGORY":
                if self.last_ingredients and self._is_style_followup(message):
                    result = await self.recommend_dishes_by_ingredients_with_style(
                        message, self.last_ingredients
                    )
                    response_text = result.get(
                        "answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다."
                    )
                    self._add_assistant_response(response_text)
                    self.last_intent = "INGREDIENTS_TO_DISHES"
                    return {
                        "answer": response_text,
                        "food_name": None,
                        "ingredients": result.get(
                            "extracted_ingredients", self.last_ingredients
                        ),
                        "recipe": [],
                    }
                else:
                    result = self.recommenders.recommend_by_category(
                        message, avoid=self.last_suggested_dishes
                    )
                    items = result.get("items", []) if isinstance(result, dict) else []
                    if (
                        not isinstance(result, dict)
                        or not items
                        or (result.get("category") == "미정")
                    ):
                        # 스타일 유도 멘트만 출력
                        response_text = "혹시 특별히 끌리는 요리 스타일(한식, 중식, 이탈리아식 등)이 있으신가요? 말씀해주시면 거기에 맞춰 맛있는 메뉴를 추천해드릴게요!"
                        # 번호 선택 방지 위해 캐시 초기화
                        self.last_suggested_dishes = []
                        self.last_suggested_turn = 0
                    else:
                        category_label = result.get("category", "한식")

                        # 디듀프: 이전 제안 및 현재 목록 내 중복 제거
                        def _norm_cat(n: str) -> str:
                            import re

                            s = (n or "").lower().strip()
                            s = re.sub(r"[\s·ㆍ・/|()-]+", "", s)
                            s = re.sub(r"[^a-z가-힣]", "", s)
                            return s

                        avoid_set = {
                            _norm_cat(x)
                            for x in getattr(self, "last_suggested_dishes", [])
                        }
                        seen = set()
                        deduped = []
                        for it in items:
                            name = (
                                it
                                if isinstance(it, str)
                                else (
                                    it.get("name", "")
                                    if isinstance(it, dict)
                                    else str(it)
                                )
                            )
                            key = _norm_cat(name)
                            if not name or key in seen or key in avoid_set:
                                continue
                            seen.add(key)
                            deduped.append(it)
                        items = deduped
                        response_text = ""
                        if category_label == "한식":
                            for i, item in enumerate(items, 1):
                                name = (
                                    item
                                    if isinstance(item, str)
                                    else item.get("name", "")
                                )
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

                    self._add_assistant_response(response_text)
                    self.last_intent = "CATEGORY"
                    # CATEGORY 결과의 카테고리를 최근 스타일로 기억하여 '다른 거' 재추천에 활용
                    if (
                        isinstance(result, dict)
                        and result.get("category")
                        and result.get("category") != "미정"
                    ):
                        self.last_style = result.get("category") or ""
                        self.last_style_ts = time.time()
                    if (result.get("category") or "") == "한식":
                        self.last_suggested_dishes = [
                            str(x).strip()
                            for x in items
                            if isinstance(x, str) and x.strip()
                        ]
                    else:
                        self.last_suggested_dishes = [
                            x.get("name", "").strip()
                            for x in items
                            if isinstance(x, dict) and x.get("name")
                        ]
                    self.last_suggested_ts = time.time()
                    self.last_suggested_turn = self.turn_idx
                    return {
                        "answer": response_text,
                        "food_name": None,
                        "ingredients": [],
                        "recipe": [],
                    }

            elif intent == "INGREDIENTS_TO_DISHES":
                result = await self.recommend_dishes_by_ingredients(message)
                response_text = result.get(
                    "answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다."
                )
                extracted_ingredients = result.get("extracted_ingredients", [])
                if extracted_ingredients:
                    self.last_ingredients = extracted_ingredients
                    self.last_ingredients_ts = time.time()
                    self.last_ingredients_turn = self.turn_idx
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": [],
                    "recipe": [],
                }

            elif intent == "RECIPE":
                dish = self._extract_dish_smart(message)
                result = (
                    self.recipes.handle_vague_dish(dish)
                    if self.recipes.is_vague_dish(dish)
                    else self.recipes.get_recipe(dish)
                )
                if result.get("type") == "vague_dish":
                    varieties = result.get("varieties", [])
                    response_text = f"어떤 {dish} 레시피를 원하시나요?\n\n"
                    for i, variety in enumerate(varieties, 1):
                        response_text += f"{i}. {variety}\n"
                    response_text += (
                        f"\n다른 원하시는 {dish} 종류가 있으시면 말씀해주세요!"
                    )
                    self._add_assistant_response(response_text)
                    self.last_suggested_dishes = [
                        str(v).strip()
                        for v in varieties
                        if isinstance(v, str) and v.strip()
                    ]
                    self.last_suggested_ts = time.time()
                    return {
                        "answer": response_text,
                        "food_name": dish,
                        "ingredients": [],
                        "recipe": [],
                    }
                else:
                    title = result.get("title", dish)
                    ingredients = result.get("ingredients", [])
                    steps = result.get("steps", [])
                    if isinstance(ingredients, list):
                        self.last_ingredients = ingredients
                    response_text = "📋 [재료]\n"
                    for i, ingredient in enumerate(ingredients, 1):
                        response_text += f"{i}. {ingredient}\n"
                    response_text += "\n👨‍🍳 [조리법]\n"
                    for i, step in enumerate(steps, 1):
                        response_text += f"{i}. {step}\n"
                    simple_answer = f"네. {title}의 레시피를 알려드릴게요."
                    self._add_assistant_response(response_text)
                    return {
                        "answer": simple_answer,
                        "food_name": title,
                        "ingredients": ingredients,
                        "recipe": steps,
                    }

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
                    if isinstance(result, list):
                        self.last_ingredients = result
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": result,
                    "recipe": [],
                }

            elif intent == "TIP":
                dish = self._extract_dish_smart(message)
                result = self.recipes.get_tips(dish)
                if not result or result == ["조리 팁을 찾을 수 없습니다"]:
                    response_text = f"죄송합니다. {dish}의 조리 팁을 찾을 수 없습니다."
                else:
                    response_text = (
                        f"네, 알겠습니다! {dish}를 더 맛있게 만드는 조리 팁입니다.\n\n"
                    )
                    response_text += "💡 [조리 팁]\n"
                    for i, tip in enumerate(result, 1):
                        response_text += f"{i}. {tip}\n"
                    response_text += (
                        f"\n{dish} 레시피나 재료도 궁금하시면 말씀해주세요!"
                    )
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": [],
                    "recipe": result,
                }

            elif intent == "SUBSTITUTE":
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                user_substitute = self._extract_explicit_substitute_name(message)
                subs = self.substitutions.get_substitutions(
                    dish, ingredient, user_substitute, message, ""
                )
                target_ing = subs.get("ingredient", ingredient or "해당 재료")
                substitute_name = subs.get("substituteName", user_substitute or "")
                candidates = subs.get("substitutes", [])
                if not candidates:
                    response_text = ""
                else:
                    if substitute_name:
                        method = (
                            candidates[0].get("method_adjustment", "") or ""
                        ).strip()
                        response_text = method
                    else:
                        lines = []
                        for i, item in enumerate(candidates, 1):
                            name = item.get("name", "")
                            amount = item.get("amount", item.get("ratio_or_amount", ""))
                            method = item.get(
                                "method_adjustment", item.get("method", "")
                            )
                            parts = [str(i) + ".", name]
                            if amount:
                                parts.append(amount)
                            if method:
                                parts.append(method)
                            line = " — ".join([p for p in parts if p])
                            if line.strip():
                                lines.append(line)
                        response_text = "\n".join(lines)
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": [target_ing],
                    "recipe": [],
                }

            elif intent == "NECESSITY":
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                result = self.substitutions.get_necessity(dish, ingredient, "")
                possible = result.get("possible", False)
                flavor_change = result.get("flavor_change", "")
                response_text = f"가능: {'예' if possible else '아니오'}"
                if flavor_change:
                    response_text += f"\n맛 변화: {flavor_change}"
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": [ingredient] if ingredient else [],
                    "recipe": [],
                }

            else:
                response_text = "요리 관련 질문을 해주세요. 레시피, 재료, 조리 팁 등 무엇이든 도와드릴 수 있어요!"
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": [],
                    "recipe": [],
                }
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            error_message = (
                "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            )
            return {
                "answer": error_message,
                "food_name": None,
                "ingredients": [],
                "recipe": [],
            }

    def _extract_dish_smart(self, message: str) -> str:
        if any(pronoun in message for pronoun in PRONOUNS):
            if self.last_dish:
                return self.last_dish
        dish = find_dish_by_pattern(message)
        if dish:
            self.last_dish = dish
            return dish
        # fallback: keep last dish or unknown
        return self.last_dish or "알 수 없는 요리"

    def _extract_ingredient_to_substitute(self, message: str) -> str:
        candidate_from_inventory = match_ingredient_from_inventory(
            message, self.last_ingredients
        )
        if candidate_from_inventory:
            return candidate_from_inventory
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
                    mapped = map_to_inventory(raw_ing, self.last_ingredients)
                    return mapped or raw_ing
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
            return {
                "answer": "재료 분석 중 오류가 발생했습니다. 다시 시도해주세요.",
                "extracted_ingredients": [],
            }
        ingredients = data.get("ingredients", [])
        dishes = data.get("dishes", [])

        # 후처리 디듀프: 이전 추천과 자체 중복 제거
        def _norm(s: str) -> str:
            import re

            s = (s or "").lower().strip()
            s = re.sub(r"[\s·ㆍ・/|()-]+", "", s)
            s = re.sub(r"[^a-z가-힣]", "", s)
            return s

        avoid_set = set()
        if getattr(self, "last_suggested_dishes", None):
            avoid_set = {_norm(x) for x in self.last_suggested_dishes}
        seen = set()
        filtered = []
        for d in dishes or []:
            name = (d.get("name") if isinstance(d, dict) else str(d)).strip()
            key = _norm(name)
            if not name or key in seen or key in avoid_set:
                continue
            seen.add(key)
            filtered.append(d)
        dishes = filtered
        if not dishes:
            return {
                "answer": "해당 재료로 만들 수 있는 요리를 찾을 수 없습니다.",
                "extracted_ingredients": ingredients,
                "food_name": None,
                "recipe": [],
            }
        response_text = f"다음 재료들로 만들 수 있는 한식 요리를 추천드려요:\n\n"
        response_text += "\n🍳 [추천 요리]\n"
        for i, dish in enumerate(dishes, 1):
            if isinstance(dish, dict):
                name = (dish.get("name") or "").strip()
                desc = (
                    dish.get("description")
                    or dish.get("note")
                    or dish.get("uses")
                    or ""
                ).strip()
                line = f"{i}. {name}" if name else f"{i}."
                if desc:
                    line += f" — {desc}"
                response_text += line + "\n"
            else:
                response_text += f"{i}. {dish}\n"
        response_text += (
            "\n원하는 요리 형식이 있으신가요? (프랑스식, 이탈리아식, 미국식 등)"
        )
        response_text += "\n또는 위 요리 중 어떤 것의 레시피를 알고 싶으시면 번호나 요리명을 말씀해주세요!"
        self.last_suggested_dishes = [
            (d.get("name") if isinstance(d, dict) else str(d)).strip()
            for d in dishes
            if (isinstance(d, dict) and d.get("name")) or isinstance(d, str)
        ]
        self.last_suggested_ts = time.time()
        self.last_suggested_turn = self.turn_idx
        return {
            "answer": response_text,
            "extracted_ingredients": ingredients,
            "recommended_dishes": dishes,
        }

    async def recommend_dishes_by_ingredients_with_style(
        self, message: str, last_ingredients: List[str]
    ) -> Dict:
        lower_msg = message.lower()
        inferred = next(
            (
                c
                for c in CUISINE_PROFILES
                if any(k in lower_msg or k in message for k in c["keywords"])
            ),
            None,
        )
        if inferred is None:
            # 스타일 키워드가 없으면, 최근 스타일이 있으면 그것을 사용하고, 없으면 명확히 요청
            if getattr(self, "last_style", ""):
                inferred = next(
                    (c for c in CUISINE_PROFILES if c["key"] == self.last_style), None
                )
            if inferred is None:
                # 선택 번호 매핑 혼선을 막기 위해 추천 캐시를 비움
                self.last_suggested_dishes = []
                self.last_suggested_turn = 0
                return {
                    "answer": "원하시는 요리 스타일을 알려주세요. (예: 프랑스식, 이탈리아식, 미국식 등)",
                    "extracted_ingredients": last_ingredients,
                }
        category_key = inferred["key"]
        chef = inferred["chef"]
        ingredients_str = ", ".join(last_ingredients)
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
        - 아래 목록과 '이름이 겹치거나 같은 계열/변형'은 제외하세요:

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
            return {
                "answer": f"{category_key} 스타일 요리 추천 중 오류가 발생했습니다. 다시 시도해주세요.",
                "extracted_ingredients": last_ingredients,
            }
        style = data.get("style", category_key)
        dishes = data.get("dishes", [])

        # 후처리 디듀프: 이전 추천과 자체 중복 제거
        def _norm(s: str) -> str:
            import re

            s = (s or "").lower().strip()
            s = re.sub(r"[\s·ㆍ・/|()-]+", "", s)
            s = re.sub(r"[^a-z가-힣]", "", s)
            return s

        avoid_set = set()
        if getattr(self, "last_suggested_dishes", None):
            avoid_set = {_norm(x) for x in self.last_suggested_dishes}
        seen = set()
        filtered = []
        for d in dishes or []:
            name = (d.get("name") if isinstance(d, dict) else str(d)).strip()
            key = _norm(name)
            if not name or key in seen or key in avoid_set:
                continue
            seen.add(key)
            filtered.append(d)
        dishes = filtered
        if not dishes:
            return {
                "answer": f"해당 재료로 만들 수 있는 {category_key} 요리를 찾을 수 없습니다.",
                "extracted_ingredients": last_ingredients,
            }
        response_text = f"{style} 스타일 추천 요리:\n\n"
        for i, dish in enumerate(dishes, 1):
            if isinstance(dish, dict):
                name = (dish.get("name") or "").strip()
                desc = (
                    dish.get("description")
                    or dish.get("note")
                    or dish.get("uses")
                    or ""
                ).strip()
                line = f"{i}. {name}" if name else f"{i}."
                if desc:
                    line += f" — {desc}"
                response_text += line + "\n"
            else:
                response_text += f"{i}. {dish}\n"
        response_text += "\n원하는 요리의 레시피를 알려드릴까요? 번호(예: 1번)나 요리명을 말씀해 주세요."
        self.last_suggested_dishes = [
            (d.get("name") if isinstance(d, dict) else str(d)).strip()
            for d in dishes
            if (isinstance(d, dict) and d.get("name")) or isinstance(d, str)
        ]
        self.last_suggested_ts = time.time()
        self.last_style = style
        self.last_style_ts = time.time()
        self.last_suggested_turn = self.turn_idx
        return {
            "answer": response_text,
            "extracted_ingredients": last_ingredients,
            "style": style,
            "recommended_dishes": dishes,
        }

    async def _handle_selection_if_any(self, message: str):
        import re

        text = (message or "").strip()
        if not text:
            return None
        if not re.search(r"\d", text):
            return None
        if not getattr(self, "last_suggested_dishes", None):
            return None
        # 직전 추천 직후의 선택만 허용하여 오래된 목록으로의 잘못된 매핑을 방지
        if (
            not getattr(self, "last_suggested_turn", 0)
            or (self.turn_idx - self.last_suggested_turn) > 1
        ):
            return None
        indices = re.findall(r"\d+", text)
        if not indices:
            return None
        unique_idxs: List[int] = []
        for s in indices:
            try:
                n = int(s)
                if (
                    n >= 1
                    and n <= len(self.last_suggested_dishes)
                    and n not in unique_idxs
                ):
                    unique_idxs.append(n)
            except Exception:
                continue
        if not unique_idxs:
            return None
        chosen_dishes = [self.last_suggested_dishes[i - 1] for i in unique_idxs]
        main_dish = chosen_dishes[0]
        recipe = self.recipes.get_recipe(main_dish)
        answer_lines: List[str] = []
        if len(chosen_dishes) > 1:
            answer_lines.append(
                "여러 개를 선택하셨네요. 먼저 1개 레시피부터 안내드릴게요. 나머지 요리도 원하시면 다시 번호를 말씀해주세요."
            )
        answer_lines.append(
            f"네. {recipe.get('title', main_dish)}의 레시피를 알려드릴게요."
        )
        self._add_assistant_response("\n".join(answer_lines))
        self.last_intent = "RECIPE"
        return {
            "answer": "\n".join(answer_lines),
            "food_name": recipe.get("title", main_dish),
            "ingredients": recipe.get("ingredients", []),
            "recipe": recipe.get("steps", recipe.get("recipe", [])),
        }
