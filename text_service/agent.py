import google.generativeai as genai
import os
import json
import logging
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
import aiohttp

# 환경 변수 로드
load_dotenv()
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEXT_SERVICE_URL = "http://localhost:8002"


class TextAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.conversation_history = []
        self.last_dish = None  # 마지막 언급된 요리명 캐시
        self.last_ingredients = []  # 마지막 조회된 재료 리스트 캐시
        self.last_intent = None  # 마지막 처리한 의도
        self.last_suggested_dishes = []  # 마지막 추천한 요리명 리스트
        self.last_ingredients_ts = 0.0
        self.last_suggested_ts = 0.0
        self.cache_ttl_sec = 300  # 5분 TTL
        self.turn_idx = 0  # 처리한 사용자 턴 수
        self.last_ingredients_turn = 0  # 재료 캐시가 갱신된 턴 인덱스
        self.last_style = ""  # 최근 스타일 키
        self.last_style_ts = 0.0


    def _add_assistant_response(self, content: str):
        self.conversation_history.append({"role": "assistant", "content": content})

    def _get_recent_context(self, count: int = 3) -> str:
        if not self.conversation_history: return "대화 히스토리가 없습니다."
        recent = self.conversation_history[-count:]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

    def _is_fresh(self, ts: float) -> bool:
        if not ts:
            return False
        return (time.time() - ts) <= self.cache_ttl_sec

    def _has_explicit_new_intent(self, message: str) -> bool:
        """사용자가 명시적으로 '새 흐름'을 요청했는지 감지: 캐시 무효화 트리거.
        예: 새로운/새 요리/새 재료/기존 말고/그냥 프랑스식/랜덤/무작위 등
        """
        text = (message or "").lower()
        keywords = [
            "새로운", "새 요리", "새 재료", "기존 말고", "그냥 프랑스식", "그냥 이탈리아식",
            "랜덤", "무작위", "다른 걸", "다른 요리", "새 추천"
        ]
        return any(k in text for k in keywords)

    def _is_other_in_same_style(self, message: str) -> bool:
        """현재 스타일 내에서 다른 것을 요청하는지(예: '다른 거 추천해줘') 판별"""
        text = (message or "").lower().strip()
        if not text:
            return False
        other_keys = ["다른 거", "다른것", "다른 요리", "다른 메뉴", "또 추천", "좀 더", "more", "another"]
        return any(k in text for k in other_keys) and bool(self.last_style) and self._is_fresh(self.last_style_ts)

    def _is_cache_valid(self) -> bool:
        """시간(≤5분) AND 턴(≤3턴) 동시 충족 시 캐시 유효"""
        time_ok = self._is_fresh(self.last_ingredients_ts)
        turn_ok = (self.turn_idx - self.last_ingredients_turn) <= 3 if self.last_ingredients_turn else False
        return time_ok and turn_ok and bool(self.last_ingredients)

    def _is_style_followup(self, message: str) -> bool:
        """메시지가 '스타일만' 요청하는 후속인지 판단.
        최근 맥락이 재료/레시피/재료목록 중 하나였고, 현재 메시지에 스타일 키워드만 있고 재료/요리 행동어가 없으면 True.
        """
        text = (message or "").lower().strip()
        if not text:
            return False
        style_keywords = [
            "한식", "중식", "일식", "프랑스", "프랑스식", "이탈리아", "이탈리아식",
            "스페인", "스페인식", "지중해", "미국", "미국식", "korean", "japanese",
            "chinese", "french", "italian", "spanish", "mediterranean", "american"
        ]
        # 스타일 키워드만 있고 재료/요리 관련 단어가 거의 없는지
        has_style = any(k in text for k in style_keywords)
        non_style_hints = ["재료", "레시피", "만들", "요리", "준비", "굽", "볶", "끓"]
        has_non_style = any(k in text for k in non_style_hints)
        recent_allows_follow = self.last_intent in {"INGREDIENTS_TO_DISHES", "RECIPE", "INGREDIENTS"}
        return has_style and not has_non_style and recent_allows_follow and bool(self.last_ingredients) and self._is_fresh(self.last_ingredients_ts)


    async def process_message(self, message: str) -> dict:
        """메인 메시지 처리 함수"""
        try:
            # 턴 증가
            self.turn_idx += 1
            # 현재 메시지를 히스토리에 추가
            self.conversation_history.append({"role": "user", "content": message})

            # 숫자 선택(예: "1번", "2,3번") 후속 처리를 최우선 감지
            selection_result = await self._handle_selection_if_any(message)
            if selection_result is not None:
                return selection_result

            # 명시 신호 우선: 새 흐름 요청이면 캐시 무효화
            if self._has_explicit_new_intent(message):
                self.last_ingredients = []
                self.last_ingredients_ts = 0.0
                self.last_ingredients_turn = 0
                self.last_suggested_dishes = []
                self.last_suggested_ts = 0.0
                logger.info("명시 신호 감지 → 캐시 무효화 후 정상 분기로 진행")

            # 동일 스타일 내 '다른 거' 요청 우선: 최근 스타일/재료가 유효하면 같은 스타일로 재추천
            if self._is_other_in_same_style(message) and self._is_cache_valid():
                logger.info(f"동일 스타일 재추천: style={self.last_style}, ingredients={self.last_ingredients}")
                result = await self.recommend_dishes_by_ingredients_with_style(self.last_style, self.last_ingredients)
                response_text = result.get("answer", "추천을 찾을 수 없습니다.")
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": result.get("extracted_ingredients", self.last_ingredients),
                    "recipe": []
                }

            # [선제 라우팅] 스타일 후속: 직전 맥락이 재료/레시피/재료목록이고, 메시지가 스타일 키워드만이면
            if self._is_style_followup(message) and self._is_cache_valid():
                logger.info("스타일-후속 선제 라우팅 실행: 최근 재료 캐시로 스타일 추천 우회")
                result = await self.recommend_dishes_by_ingredients_with_style(message, self.last_ingredients)
                response_text = result.get("answer", "해당 재료로 만들 수 있는 요리를 찾을 수 없습니다.")
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": result.get("extracted_ingredients", self.last_ingredients),
                    "recipe": []
                }
            
            # 의도 분류 (최적화된 단일 호출)
            intent = await self.classify_intent_optimized(message)
            logger.info(f"의도 분류 결과: {intent}")

            # 의도별 처리
            if intent == "CATEGORY":
                # 최근 재료가 있더라도, 순수 카테고리 요청인지/스타일 후속 요청인지 구분
                if self.last_ingredients and self._is_style_followup(message):
                    logger.info(
                        f"CATEGORY → 스타일 후속 요청 감지. 최근 재료로 우회: {self.last_ingredients}"
                    )
                    result = await self.recommend_dishes_by_ingredients_with_style(message, self.last_ingredients)
                    response_text = result.get("answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다.")
                    self._add_assistant_response(response_text)
                    self.last_intent = "INGREDIENTS_TO_DISHES"
                    return {
                        "answer": response_text,
                        "food_name": None,
                        "ingredients": result.get("extracted_ingredients", self.last_ingredients),
                        "recipe": []
                    }
                else:
                    # 기존 카테고리 로직
                    result = await self.recommend_dishes_optimized(message)
                    # 결과 표준화: {category: "한식"|..., items: [...]}
                    if not isinstance(result, dict) or not result.get("items"):
                        response_text = "죄송합니다. 추천 요리를 찾을 수 없습니다. 다른 카테고리를 말씀해주세요."
                    else:
                        category_label = result.get("category", "한식")
                        items = result.get("items", [])
                        response_text = ""
                        if category_label == "한식":
                            # 한식: 요리명만
                            for i, item in enumerate(items, 1):
                                name = item if isinstance(item, str) else item.get("name", "")
                                if name:
                                    response_text += f"{i}. {name}\n"
                        else:
                            # 타국 요리: 간단한 설명 포함
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
                    
                    self._add_assistant_response(response_text)
                    self.last_intent = "CATEGORY"
                    # 번호 선택 캐시 저장(한식은 단일 문자열 배열, 그 외는 name 추출)
                    if category_label == "한식":
                        self.last_suggested_dishes = [str(x).strip() for x in items if isinstance(x, str) and x.strip()]
                    else:
                        self.last_suggested_dishes = [x.get("name", "").strip() for x in items if isinstance(x, dict) and x.get("name")]
                    self.last_suggested_ts = time.time()
                    return {
                        "answer": response_text,
                        "food_name": None,
                        "ingredients": [],
                        "recipe": []
                    }
                
            elif intent == "INGREDIENTS_TO_DISHES":
                result = await self.recommend_dishes_by_ingredients(message)
                response_text = result.get("answer", "재료로 만들 수 있는 요리를 찾을 수 없습니다.")
                
                # 추출된 재료를 last_ingredients에 저장
                extracted_ingredients = result.get("extracted_ingredients", [])
                if extracted_ingredients:
                    self.last_ingredients = extracted_ingredients
                    self.last_ingredients_ts = time.time()
                    self.last_ingredients_turn = self.turn_idx
                    logger.info(f"재료 캐시 업데이트: {self.last_ingredients}")
                
                self._add_assistant_response(response_text)
                self.last_intent = "INGREDIENTS_TO_DISHES"
                return {
                    "answer": response_text,
                    "food_name": None,
                    # 프론트에서 레시피 카드로 렌더링되지 않도록 비움
                    "ingredients": [],
                    "recipe": []
                }
                
            elif intent == "RECIPE":
                dish = self._extract_dish_smart(message)
                result = await self.get_recipe_optimized(dish)
                
                if result.get("type") == "vague_dish":
                    varieties = result.get("varieties", [])
                    response_text = f"어떤 {dish} 레시피를 원하시나요?\n\n"
                    for i, variety in enumerate(varieties, 1):
                        response_text += f"{i}. {variety}\n"
                    response_text += f"\n다른 원하시는 {dish} 종류가 있으시면 말씀해주세요!"
                    
                    self._add_assistant_response(response_text)
                    # 번호 선택 캐시 저장(모호한 하위 종류)
                    self.last_suggested_dishes = [str(v).strip() for v in varieties if isinstance(v, str) and v.strip()]
                    self.last_suggested_ts = time.time()
                    return {
                        "answer": response_text,
                        "food_name": dish,
                        "ingredients": [],
                        "recipe": []
                    }
                else:
                    title = result.get("title", dish)
                    ingredients = result.get("ingredients", [])
                    steps = result.get("steps", [])

                    # 최근 재료 캐시 업데이트
                    if isinstance(ingredients, list):
                        self.last_ingredients = ingredients
                    
                    # 레시피만 명확하게 출력 (불필요 문구 제거)
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
                        "recipe": steps
                    }
                
            elif intent == "INGREDIENTS":
                dish = self._extract_dish_smart(message)
                result = await self.get_ingredients_optimized(dish)
                
                if not result or result == ["재료 정보를 찾을 수 없습니다"]:
                    response_text = "재료 정보를 찾을 수 없습니다"
                else:
                    # 간결한 출력: 제목/도입 없이 재료만 나열
                    response_lines = []
                    for i, ingredient in enumerate(result, 1):
                        response_lines.append(f"{i}. {ingredient}")
                    response_text = "\n".join(response_lines)

                    # 최근 재료 캐시 업데이트
                    if isinstance(result, list):
                        self.last_ingredients = result
                
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": result,
                    "recipe": []
                }
                
            elif intent == "TIP":
                dish = self._extract_dish_smart(message)
                result = await self.get_tips_optimized(dish)
                
                if not result or result == ["조리 팁을 찾을 수 없습니다"]:
                    response_text = f"죄송합니다. {dish}의 조리 팁을 찾을 수 없습니다."
                else:
                    response_text = f"네, 알겠습니다! {dish}를 더 맛있게 만드는 조리 팁입니다.\n\n"
                    response_text += "💡 [조리 팁]\n"
                    for i, tip in enumerate(result, 1):
                        response_text += f"{i}. {tip}\n"
                    response_text += f"\n{dish} 레시피나 재료도 궁금하시면 말씀해주세요!"
                
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": [],
                    "recipe": result
                }
            
            elif intent == "SUBSTITUTE":
                # 재료 대체 요청 처리
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                user_substitute = self._extract_explicit_substitute_name(message)
                subs = await self.get_substitutions_optimized(dish, ingredient, user_substitute, message)
                target_ing = subs.get("ingredient", ingredient or "해당 재료")
                substitute_name = subs.get("substituteName", user_substitute or "")
                candidates = subs.get("substitutes", [])
                
                if not candidates:
                    response_text = ""
                else:
                    if substitute_name:
                        # 사용자가 대체 재료를 명시했을 때: 한 줄의 조리 방법 수정만 출력
                        method = candidates[0].get("method_adjustment", "").strip()
                        response_text = method
                    else:
                        # 이름 / 양 / 조리 방법 수정만 출력 (3개)
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
                
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": dish,
                    "ingredients": [target_ing],
                    "recipe": []
                }
            
            elif intent == "NECESSITY":
                # 재료 필요 여부 처리 (가능 여부 + 맛 변화만)
                dish = self._extract_dish_smart(message)
                ingredient = self._extract_ingredient_to_substitute(message)
                result = await self.get_ingredient_necessity(dish, ingredient)
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
                    "recipe": []
                }
                
            else:
                response_text = "요리 관련 질문을 해주세요. 레시피, 재료, 조리 팁 등 무엇이든 도와드릴 수 있어요!"
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "food_name": None,
                    "ingredients": [],
                    "recipe": []
                }
                
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            error_message = "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            return {
                "answer": error_message,
                "food_name": None,
                "ingredients": [],
                "recipe": []
            }

    async def classify_intent_optimized(self, message: str) -> str:
        """최적화된 의도 분류"""
        context = self._get_recent_context(5)  # 최근 3개 메시지만 사용
        
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
            resp = self.model.generate_content(prompt)
            intent = resp.text.strip().upper()
            logger.debug(f"의도 분류 프롬프트: {prompt}")
            logger.debug(f"의도 분류 응답: {intent}")
            return intent
        except Exception as e:
            logger.error(f"의도 분류 오류: {e}")
            return "OTHER"

    def _extract_dish_smart(self, message: str) -> str:
        """스마트 요리명 추출"""
        # 대명사 처리
        pronouns = ["그거", "그 음식", "이거", "저거", "그것", "이것"]
        if any(pronoun in message for pronoun in pronouns):
            if self.last_dish:
                logger.info(f"대명사 감지, 이전 요리명 사용: {self.last_dish}")
                return self.last_dish
        
        # 정규식으로 요리명 찾기
        dish = self._find_dish_by_pattern(message)
        if dish:
            self.last_dish = dish
            logger.info(f"패턴으로 요리명 추출: {dish}")
            return dish
        
        # LLM으로 요리명 추출 (최적화)
        dish = self._extract_dish_with_llm(message)
        if dish and dish != "알 수 없는 요리":
            self.last_dish = dish
            logger.info(f"LLM으로 요리명 추출: {dish}")
            return dish
        
        # 기본값
        return self.last_dish or "알 수 없는 요리"

    def _find_dish_by_pattern(self, message: str) -> str:
        """정규식 패턴으로 요리명 찾기"""
        import re
        
        patterns = [
            r"([가-힣]+)\s+재료",
            r"([가-힣]+)\s+레시피",
            r"([가-힣]+)\s+만드는\s+법",
            r"([가-힣]+)\s+조리법",
            r"([가-힣]+)\s+팁"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                dish = match.group(1)
                if len(dish) > 1 and len(dish) < 20:
                    return dish
        return ""

    def _extract_dish_with_llm(self, message: str) -> str:
        """LLM으로 요리명 추출 (최적화)"""
        context = self._get_recent_context(2)
        
        prompt = f"""
        대화: {context}
        현재: {message}
        
        위 대화에서 언급된 요리명을 찾아서 출력하세요.
        요리명만 출력하세요.
        """
        
        try:
            resp = self.model.generate_content(prompt)
            dish_raw = resp.text.strip()
            # 여러 개가 나오는 경우(줄바꿈/쉼표/연결 어미) 첫 번째로 적합한 하나만 선택
            import re
            candidates = [
                c.strip() for c in re.split(r"[\n,/]|\s*(?:와|과|랑|그리고|및)\s*", dish_raw)
                if c and len(c.strip()) < 50
            ]
            # 입력 메시지와 가장 유사한 후보를 선택
            if candidates:
                matched = next((c for c in candidates if c and c in message), None)
                chosen = matched or candidates[0]
            else:
                chosen = dish_raw
            dish = chosen.strip()
            return dish if dish and len(dish) < 50 else ""
        except Exception as e:
            logger.error(f"LLM 요리명 추출 오류: {e}")
            return ""

    async def recommend_dishes_optimized(self, message: str) -> dict:
        """최적화된 요리 추천: 출력 형식은 카테고리별 다름"""
        # 카테고리별 담당 셰프 및 키워드 매핑
        cuisine_profiles = [
            {"key": "한식", "chef": "강레오, 안성재", "keywords": ["한식", "korean", "코리안"]},
            {"key": "중식", "chef": "Ken Hom", "keywords": ["중식", "중국", "차이니즈"]},
            {"key": "일식", "chef": "Yoshihiro Murata", "keywords": ["일식", "일본", "재패니즈", "japanese", "japan"]},
            {"key": "프랑스식", "chef": "Pierre Koffmann", "keywords": ["프랑스", "프랑스식", "프렌치", "french"]},
            {"key": "이탈리아식", "chef": "Massimo Bottura", "keywords": ["이탈리아", "이탈리아식", "이탈리안", "italian"]},
            {"key": "스페인식", "chef": "José Andrés", "keywords": ["스페인", "스페인식", "spanish"]},
            {"key": "지중해식", "chef": "Yotam Ottolenghi", "keywords": ["지중해", "mediterranean"]},
            {"key": "미국식", "chef": "Gordon Ramsay", "keywords": ["미국", "미국식", "아메리칸", "american"]},
        ]
        lower_msg = message.lower()
        inferred = next((c for c in cuisine_profiles if any(k in lower_msg or k in message for k in c["keywords"])), None)
        # 모호하면 기본값으로 특정 카테고리를 추정하지 않음
        if inferred is None:
            logger.info("카테고리 모호: 명시적 국가/스타일이 없어 확인 필요")
            return {"category": "미정", "items": []}
        category_key = inferred["key"]
        chef = inferred["chef"]

        # 프롬프트: 한식과 그 외의 출력 포맷 분리
        if category_key == "한식":
            prompt = f"""
            당신은 {chef} 셰프입니다.
            사용자가 명시적으로 다른 나라 요리를 지정하지 않았다면 기본적으로 한식을 추천하세요.
            집에서 쉽게 만들 수 있는 가정식/홈스타일 요리를 우선 추천하고, '요리 욕구를 자극하는' 메뉴로 선정하세요.
            조리 난이도는 쉬움~보통, 준비 시간은 15~40분 내 위주로 구성하세요.
            한식 요리 5개를 JSON 배열로만 출력하세요(요리명만 출력).
            예시: ["김치찌개", "된장찌개", "불고기", "비빔밥", "잡채"]
            """
        else:
            prompt = f"""
            당신은 {chef} 셰프입니다.
            요청 카테고리: {category_key}
            집에서 쉽게 만들 수 있는 홈스타일 버전으로 추천하세요(구하기 쉬운 재료/기본 도구).
            조리 난이도는 쉬움~보통, 준비 시간은 15~40분 내 위주.
            {category_key} 요리 5개를 JSON 배열로만 출력하되, 각 항목은 이름과 한 줄 설명을 포함하세요.
            출력 형식 예시:
            [
              {{"name": "요리명1", "description": "간단한 한 줄 설명"}},
              {{"name": "요리명2", "description": "간단한 한 줄 설명"}}
            ]
            """

        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"CATEGORY JSON 파싱 실패. 응답 미리보기: {response_text[:200]}")
                return {"category": category_key, "items": []}
            items: list
            if category_key == "한식":
                # 기대 형식: ["요리1", ...]
                items = data if isinstance(data, list) else []
                # 안전장치: 문자열만 유지
                items = [x for x in items if isinstance(x, str) and x.strip()]
            else:
                # 기대 형식: [{name, description}, ...]
                items = data if isinstance(data, list) else []
                normalized = []
                for it in items:
                    if isinstance(it, dict) and it.get("name"):
                        normalized.append({
                            "name": it.get("name", "").strip(),
                            "description": it.get("description", "").strip()
                        })
                    elif isinstance(it, str):
                        normalized.append({"name": it.strip(), "description": ""})
                items = [x for x in normalized if x.get("name")]

            return {"category": category_key, "items": items}
        except Exception as e:
            logger.error(f"요리 추천 오류: {e}")
            return {"category": category_key, "items": []}

    def _extract_ingredient_to_substitute(self, message: str) -> str:
        """사용자 메시지에서 대체하려는 '대상 재료'를 추출하되, 최근 레시피 재료 리스트와 교차 확인"""
        import re

        # 1) 최근 재료 리스트에서 직접 매칭 시도
        candidate_from_inventory = self._match_ingredient_from_inventory(message)
        if candidate_from_inventory:
            return candidate_from_inventory

        # 2) 패턴 기반 추출 시도
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
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                raw_ing = match.group(1).strip()
                if 1 < len(raw_ing) < 30:
                    # 인벤토리에 있으면 그 이름으로 보정
                    mapped = self._map_to_inventory(raw_ing)
                    return mapped or raw_ing

        # 3) 실패 시 LLM 시도
        llm_name = self._extract_ingredient_with_llm(message)
        if llm_name:
            mapped = self._map_to_inventory(llm_name)
            return mapped or llm_name
        return ""

    def _normalize_ingredient_name(self, text: str) -> str:
        """재료명에서 수량/단위를 제거하고 핵심 명사만 남김"""
        import re
        if not isinstance(text, str):
            return ""
        name = text
        # 괄호 및 그 안의 수량 제거
        name = re.sub(r"\([^)]*\)", "", name)
        # 숫자/기호 제거
        name = re.sub(r"[0-9]+\s*[gGkKmMlL컵tspTB]+", "", name)
        name = re.sub(r"[0-9.,/%]+", "", name)
        # 불필요 공백 정리
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _tokenize_korean_phrase(self, text: str) -> list:
        """간단 토크나이저: 공백 기준, 한글/영문만 유지"""
        import re
        if not text:
            return []
        tokens = re.findall(r"[가-힣A-Za-z]+", text)
        return [t for t in tokens if len(t) >= 2]

    def _match_ingredient_from_inventory(self, message: str) -> str:
        """최근 재료 목록에서 사용자 메시지에 등장하는 재료를 탐색하여 가장 그럴듯한 것을 반환"""
        if not self.last_ingredients:
            return ""
        message_text = message or ""
        message_tokens = set(self._tokenize_korean_phrase(message_text))
        if not message_tokens:
            return ""

        best_match = (0, "")  # (점수, 원본 재료 문자열)
        for ing in self.last_ingredients:
            base = self._normalize_ingredient_name(ing)
            if not base:
                continue
            base_tokens = self._tokenize_korean_phrase(base)
            if not base_tokens:
                continue

            # 토큰 매칭 점수: 메시지에 등장하는 토큰 수와 길이 기반 가중치
            matched_tokens = [t for t in base_tokens if t in message_tokens or t in message_text]
            if not matched_tokens:
                # 길이가 긴 베이스명이 메시지 내 부분문자열로 있으면 가점
                if base and base in message_text:
                    score = len(base)
                else:
                    score = 0
            else:
                score = sum(len(t) for t in matched_tokens)

            if score > best_match[0]:
                best_match = (score, ing)

        return best_match[1]

    def _map_to_inventory(self, name: str) -> str:
        """간단 매핑: 추출된 이름을 최근 재료 목록의 가장 근접한 항목으로 보정"""
        if not name or not self.last_ingredients:
            return ""
        name_norm = self._normalize_ingredient_name(name)
        message_like = name_norm
        # 가장 긴 부분문자열 일치 기준으로 선택
        best = (0, "")
        for ing in self.last_ingredients:
            base = self._normalize_ingredient_name(ing)
            if not base:
                continue
            score = 0
            if message_like in base or base in message_like:
                score = min(len(message_like), len(base))
            elif message_like and base and (message_like in ing or base in name):
                score = min(len(message_like), len(base)) // 2
            if score > best[0]:
                best = (score, ing)
        return best[1]

    def _extract_explicit_substitute_name(self, message: str) -> str:
        """사용자가 제시한 대체 재료명 추출 (예: X 말고 Y 써도 돼?)"""
        import re
        # 패턴: "X 말고 Y", "X 대신 Y", "X 빼고 Y", 등에서 Y를 추출
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
        # 실패 시 LLM 시도
        context = self._get_recent_context(2)
        prompt = f"""
        대화: {context}
        현재: {message}
        사용자가 제시한 대체 재료가 있다면 그 이름만 출력하세요. 없다면 빈 문자열.
        """
        try:
            resp = self.model.generate_content(prompt)
            name = resp.text.strip()
            return name if name and len(name) < 50 and name != "없음" else ""
        except Exception:
            return ""

    def _extract_ingredient_with_llm(self, message: str) -> str:
        """LLM을 이용해 대체 대상 재료명 추출"""
        context = self._get_recent_context(2)
        prompt = f"""
        대화: {context}
        현재: {message}
        사용자가 대체를 원하는 재료명을 1개만 출력하세요. 불명확하면 가장 가능성이 높은 것 1개만.
        재료명만 출력하세요.
        """
        try:
            resp = self.model.generate_content(prompt)
            name = resp.text.strip()
            return name if name and len(name) < 50 else ""
        except Exception:
            return ""

    async def get_substitutions_optimized(self, dish: str, ingredient: str, user_substitute: str, message: str) -> dict:
        """재료 대체안 추천"""
        target = ingredient or "핵심 재료"
        context = self._get_recent_context(3)
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
        try:
            resp = self.model.generate_content(prompt)
            response_text = resp.text.strip()
            data = json.loads(response_text)
            # 안전장치
            data.setdefault("ingredient", target)
            data.setdefault("substituteName", user_substitute or "")
            data.setdefault("substitutes", [])
            return data
        except Exception as e:
            logger.error(f"재료 대체 추천 오류: {e}")
            return {"ingredient": target, "substituteName": user_substitute or "", "substitutes": []}

    async def get_ingredient_necessity(self, dish: str, ingredient: str) -> dict:
        """재료 필요 여부: 가능여부 + 맛 변화만"""
        target = ingredient or "핵심 재료"
        context = self._get_recent_context(3)
        prompt = f"""
        당신은 프로 요리사입니다.
        대화 맥락: {context}
        요리: '{dish}'
        재료: '{target}'
        
        질문: '{target}'이(가) 반드시 필요한가?
        JSON으로만 출력하세요. 필드는 possible(불리언), flavor_change(문장 1줄)만 포함하세요. 다른 필드/설명은 금지.
        예시: {{"possible": true, "flavor_change": "감칠맛이 약간 줄어듭니다"}}
        """
        try:
            resp = self.model.generate_content(prompt)
            response_text = resp.text.strip()
            data = json.loads(response_text)
            return {
                "possible": bool(data.get("possible", False)),
                "flavor_change": str(data.get("flavor_change", ""))
            }
        except Exception as e:
            logger.error(f"재료 필요 여부 판단 오류: {e}")
            return {"possible": False, "flavor_change": ""}
    
    async def recommend_dishes_by_ingredients(self, message: str) -> dict:
        """재료로 요리 추천 (개선된 버전)"""
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
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON 파싱 실패: {json_err}. 정리된 텍스트: {response_text[:200]}")
                return {
                    "answer": "재료 분석 중 오류가 발생했습니다. 다시 시도해주세요.",
                    "extracted_ingredients": []
                }
            
            ingredients = result.get("ingredients", [])
            dishes = result.get("dishes", [])
            
            if not dishes:
                return {
                    "answer": "해당 재료로 만들 수 있는 요리를 찾을 수 없습니다.",
                    "extracted_ingredients": ingredients,
                    "food_name": None,
                    "recipe": []
                }
            
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
            
            # 최근 추천 요리 캐시(번호 선택 후속 처리용)
            self.last_suggested_dishes = [
                (d.get("name") if isinstance(d, dict) else str(d)).strip()
                for d in dishes
                if (isinstance(d, dict) and d.get("name")) or isinstance(d, str)
            ]
            self.last_suggested_ts = time.time()

            return {
                "answer": response_text,
                "extracted_ingredients": ingredients,
                "recommended_dishes": dishes
            }
            
        except Exception as e:
            logger.error(f"재료 기반 요리 추천 오류: {e}")
            return {
                "answer": "재료 분석 중 오류가 발생했습니다. 다시 시도해주세요.",
                "extracted_ingredients": []
            }

    async def recommend_dishes_by_ingredients_with_style(self, message: str, last_ingredients: list) -> dict:
        """기존 재료로 다른 국가 스타일 요리 추천"""
        # 요청된 요리 스타일 추출
        cuisine_profiles = [
            {"key": "한식", "chef": "강레오, 안성재", "keywords": ["한식", "korean", "코리안"]},
            {"key": "중식", "chef": "Ken Hom", "keywords": ["중식", "중국", "차이니즈"]},
            {"key": "일식", "chef": "Yoshihiro Murata", "keywords": ["일식", "일본", "재패니즈", "japanese", "japan"]},
            {"key": "프랑스식", "chef": "Pierre Koffmann", "keywords": ["프랑스", "프랑스식", "프렌치", "french"]},
            {"key": "이탈리아식", "chef": "Massimo Bottura", "keywords": ["이탈리아", "이탈리아식", "이탈리안", "italian"]},
            {"key": "스페인식", "chef": "José Andrés", "keywords": ["스페인", "스페인식", "spanish"]},
            {"key": "지중해식", "chef": "Yotam Ottolenghi", "keywords": ["지중해", "mediterranean"]},
            {"key": "미국식", "chef": "Gordon Ramsay", "keywords": ["미국", "미국식", "아메리칸", "american"]},
        ]
        
        lower_msg = message.lower()
        inferred = next((c for c in cuisine_profiles if any(k in lower_msg or k in message for k in c["keywords"])), None)
        # 모호하면 기본 한식
        if inferred is None:
            inferred = cuisine_profiles[0]
        
        category_key = inferred["key"]
        chef = inferred["chef"]
        
        # 기존 재료를 문자열로 변환
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
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            result = json.loads(response_text)
            style = result.get("style", category_key)
            dishes = result.get("dishes", [])
            
            if not dishes:
                return {
                    "answer": f"해당 재료로 만들 수 있는 {category_key} 요리를 찾을 수 없습니다.",
                    "extracted_ingredients": last_ingredients
                }
            
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
            
            # 최근 추천 요리 캐시(번호 선택 후속 처리용)
            self.last_suggested_dishes = [
                (d.get("name") if isinstance(d, dict) else str(d)).strip()
                for d in dishes
                if (isinstance(d, dict) and d.get("name")) or isinstance(d, str)
            ]
            self.last_suggested_ts = time.time()
            self.last_style = style
            self.last_style_ts = time.time()

            return {
                "answer": response_text,
                "extracted_ingredients": last_ingredients,
                "style": style,
                "recommended_dishes": dishes
            }
            
        except Exception as e:
            logger.error(f"스타일별 재료 기반 요리 추천 오류: {e}")
            return {
                "answer": f"{category_key} 스타일 요리 추천 중 오류가 발생했습니다. 다시 시도해주세요.",
                "extracted_ingredients": last_ingredients
            }

    async def get_recipe_optimized(self, dish: str) -> dict:
        """최적화된 레시피 조회"""
        if self._is_vague_dish(dish):
            return await self._handle_vague_dish_optimized(dish)
        
        prompt = f"""
        당신은 세계적으로 유명한 프로 셰프입니다.
        Pierre Koffmann(프랑스), Gordon Ramsay(미국식), Ken Hom(중식), Massimo Bottura(이탈리아), José Andrés(스페인식), Yotam Ottolenghi(지중해식), 강레오(한식), 안성재(한식) 셰프의 경험을 바탕으로 정확한 레시피를 제공합니다.
        
        '{dish}' 레시피를 JSON으로 작성하세요. '{dish}'가 요리명이 아닐 경우, 메시지 전체에서 가장 가능성이 높은 **단일 요리명 1개**를 추론해 그 레시피만 작성하세요:
        - 조리법은 최대 15단계 이하로 작성
        - 복잡한 과정은 요약해서 핵심만 포함
        - 정확한 재료와 실용적인 조리법 제공
        - 레시피만 명확히 출력하며 팁/설명/도입문 등 불필요한 텍스트는 포함하지 마세요
        - 출력 텍스트(재료/단계 포함)에는 어떤 셰프의 이름이나 스타일/출처도 언급하지 마세요
        - 재료 형식 엄격 규칙:
          1) ingredients는 객체 배열이어야 합니다. 각 객체는 item/amount/unit 세 필드를 반드시 포함합니다.
          2) item: 재료명만 기재하세요. 손질/상태/형용사(예: 신선한, 다진, 손질된, 편썬 등)와 브랜드/원산지 정보는 제외합니다.
          3) amount: 수량 숫자만 기재하세요(정수, 소수, 분수 허용. 예: 1, 0.5, 1/2). 기호나 단위는 제외합니다. 불명확하면 빈 문자열.
          4) unit: g, ml, 컵, 큰술, 작은술, 개, 마리, 통, 쪽, 톨 등 단위를 한글/영문 단위명으로만 기재하세요. 단위가 없으면 빈 문자열.
          5) 범위 표기는 최소한의 숫자만 사용(예: 1~2개 → amount="1-2", unit="개").
        
        {{
          "title": "{dish}",
          "ingredients": [{{"item": "재료명", "amount": "숫자만", "unit": "단위"}}],
          "steps": ["1단계 설명", "2단계 설명"]
        }}
        """
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            recipe = json.loads(response_text)
            
            # 조리법 단계 수 제한 (15단계 이하)
            if "steps" in recipe and len(recipe["steps"]) > 15:
                recipe["steps"] = recipe["steps"][:15]
            
            # 기본값 설정
            recipe.setdefault("title", dish)
            recipe.setdefault("ingredients", ["재료 정보를 찾을 수 없습니다"])
            recipe.setdefault("steps", ["조리법 정보를 찾을 수 없습니다"])
            
            return recipe
            
        except Exception as e:
            logger.error(f"레시피 조회 오류: {e}")
            # JSON 파싱 실패 시 자연어 응답에서 정보 추출
            return self._parse_recipe_from_text(dish, resp.text if 'resp' in locals() else "")

    async def get_ingredients_optimized(self, dish: str) -> list:
        """최적화된 재료 조회"""
        prompt = f"""
        당신은 세계적으로 유명한 프로 셰프입니다.
        Pierre Koffmann(프랑스), Gordon Ramsay(미국식), Ken Hom(중식), Massimo Bottura(이탈리아), José Andrés(스페인식), Yotam Ottolenghi(지중해식), 강레오(한식), 안성재(한식) 셰프의 전문 지식을 바탕으로 정확한 재료 정보를 제공합니다.
        
        '{dish}'에 필요한 정확한 재료와 양을 JSON 객체 배열로만 출력하세요. '{dish}'가 애매하면 메시지에서 가장 가능성이 높은 **단일 요리명 1개**를 추론해 그 재료만 출력하세요.
        각 원소는 다음 형식의 객체여야 합니다: {{"item": 재료명만, "amount": 숫자만, "unit": 단위만}}
        - item: 수식어/브랜드/원산지/손질 상태를 제외한 재료명만
        - amount: 수량 숫자만(정수/소수/분수). 불명확하면 빈 문자열
        - unit: g, ml, 컵, 큰술, 작은술, 개, 마리, 통, 쪽, 톨 등 단위명. 없으면 빈 문자열
        기타 텍스트, 코드블록, 설명은 출력하지 마세요.
        예시: [{{"item": "재료1", "amount": "100", "unit": "g"}}, {{"item": "재료2", "amount": "1/2", "unit": "컵"}}]
        """
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            ingredients = json.loads(response_text)
            
            if isinstance(ingredients, list) and len(ingredients) > 0:
                return ingredients
            else:
                return ["재료 정보를 찾을 수 없습니다"]
                
        except Exception as e:
            logger.error(f"재료 조회 오류: {e}")
            # JSON 파싱 실패 시 자연어 응답에서 재료 추출
            return self._parse_ingredients_from_text(dish, resp.text if 'resp' in locals() else "")

    async def get_tips_optimized(self, dish: str) -> list:
        """최적화된 조리 팁 조회"""
        prompt = f"""
        당신은 세계적으로 유명한 프로 셰프입니다.
        Pierre Koffmann, Gordon Ramsay, Ken Hom, Massimo Bottura, José Andrés, Yotam Ottolenghi, 강레오, 안성재 셰프의 실무 경험을 바탕으로 전문적이고 실용적인 조리 팁을 제공합니다.
        
        '{dish}'를 더 맛있게 만드는 실용적인 조리 팁 3개를 JSON 배열로 출력하세요.
        각 팁은 구체적이고 실용적이어야 하며, 셰프의 이름이나 출처는 언급하지 마세요.
        예시: ["구체적인 팁1", "실용적인 팁2", "전문가 팁3"]
        """
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            tips = json.loads(response_text)
            
            if isinstance(tips, list) and len(tips) > 0:
                return tips
            else:
                return ["조리 팁을 찾을 수 없습니다"]
                
        except Exception as e:
            logger.error(f"조리 팁 조회 오류: {e}")
            # JSON 파싱 실패 시 자연어 응답에서 팁 추출
            return self._parse_tips_from_text(dish, resp.text if 'resp' in locals() else "")

    def _is_vague_dish(self, dish: str) -> bool:
        """모호한 요리인지 확인"""
        vague_dishes = [
            "파스타", "볶음밥", "커리", "샐러드", "스테이크", "피자", 
            "라면", "국수", "밥", "면", "탕", "찌개", "볶음", "구이"
        ]
        return dish in vague_dishes

    async def _handle_vague_dish_optimized(self, dish: str) -> dict:
        """최적화된 모호한 요리 처리"""
        prompt = f"""
        당신은 세계적인 프로 셰프입니다.
        사용자가 입력한 '{dish}'가 광범위한 요리 종류라면 해당 음식의 대표적인 하위 요리 3~5가지를 JSON 배열로 출력하세요.
        요리명만 출력하고 설명은 필요없습니다.
        
        예시: ["구체적인 요리명1", "구체적인 요리명2", "구체적인 요리명3"]
        """
        
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = resp.text.strip()
            varieties = json.loads(response_text)
            
            if isinstance(varieties, list) and len(varieties) > 0:
                return {
                    "title": f"{dish} 종류 추천",
                    "varieties": varieties,
                    "type": "vague_dish"
                }
            else:
                return {
                    "title": dish,
                    "type": "vague_dish"
                }
                
        except Exception as e:
            logger.error(f"모호한 요리 처리 오류: {e}")
            return {
                "title": dish,
                "type": "vague_dish"
            }





    def _parse_recipe_from_text(self, dish: str, text: str) -> dict:
        """자연어 응답에서 레시피 정보 추출"""
        import re
        
        # 기본 구조
        recipe = {
            "title": dish,
            "ingredients": [],
            "steps": []
        }
        
        # 재료 추출 (📋, 재료, • 등으로 시작하는 부분)
        ingredient_patterns = [
            r"📋\s*재료[:\s]*([\s\S]*?)(?=👨‍🍳|조리법|👨|🍳|$)",
            r"재료[:\s]*([\s\S]*?)(?=조리법|👨‍🍳|👨|🍳|$)",
            r"•\s*([^•\n]*?)(?=\n|$)",
            r"재료[:\s]*([\s\S]*?)(?=\n\n|\n\d+\.|$)"
        ]
        
        for pattern in ingredient_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    # 중괄호 형태의 재료 파싱
                    brace_matches = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'\}", match)
                    if brace_matches:
                        for name, amount in brace_matches:
                            recipe["ingredients"].append(f"{name} {amount}")
                    else:
                        # 일반적인 재료 형태
                        lines = match.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('•') and '재료' not in line:
                                recipe["ingredients"].append(line)
                break
        
        # 조리법 추출 (👨‍🍳, 조리법, 숫자로 시작하는 부분)
        step_patterns = [
            r"👨‍🍳\s*조리법[:\s]*([\s\S]*?)(?=\n\n|$)",
            r"조리법[:\s]*([\s\S]*?)(?=\n\n|$)",
            r"(\d+\.\s*[^\n]+(?:\n\d+\.\s*[^\n]+)*)"
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    lines = match.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and re.match(r'\d+\.', line):
                            # 중복된 숫자 제거 (예: "1. 1. 닭고기를..." -> "닭고기를...")
                            step = re.sub(r'^\d+\.\s*\d+\.\s*', '', line)
                            # 단일 숫자 제거
                            step = re.sub(r'^\d+\.\s*', '', step)
                            if step:
                                recipe["steps"].append(step)
                break
        
        # 조리법 단계 수 제한 (15단계 이하)
        if len(recipe["steps"]) > 15:
            recipe["steps"] = recipe["steps"][:15]
        
        # 기본값 설정
        if not recipe["ingredients"]:
            recipe["ingredients"] = ["재료 정보를 찾을 수 없습니다"]
        if not recipe["steps"]:
            recipe["steps"] = ["조리법 정보를 찾을 수 없습니다"]
        
        return recipe

    def _parse_ingredients_from_text(self, dish: str, text: str) -> list:
        """자연어 응답에서 재료 정보 추출"""
        import re
        
        ingredients = []
        
        # 중괄호 형태의 재료 파싱 (예: {'name': '닭고기', 'amount': '1kg', 'unit': 'kg'})
        brace_matches = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'(?:,\s*'unit':\s*'([^']+)')?\}", text)
        if brace_matches:
            for match in brace_matches:
                name = match[0]
                amount = match[1]
                unit = match[2] if len(match) > 2 and match[2] else ""
                if unit and unit != "''":
                    ingredients.append(f"{name} {amount}{unit}")
                else:
                    ingredients.append(f"{name} {amount}")
            return ingredients
        
        # 기존 중괄호 형태 (unit 없음)
        brace_matches_old = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'\}", text)
        if brace_matches_old:
            for name, amount in brace_matches_old:
                ingredients.append(f"{name} {amount}")
            return ingredients
        
        # 일반적인 재료 형태 파싱
        ingredient_patterns = [
            r"📋\s*재료[:\s]*([\s\S]*?)(?=👨‍🍳|조리법|👨|🍳|$)",
            r"재료[:\s]*([\s\S]*?)(?=조리법|👨‍🍳|👨|🍳|$)",
            r"•\s*([^•\n]*?)(?=\n|$)",
            r"재료[:\s]*([\s\S]*?)(?=\n\n|\n\d+\.|$)"
        ]
        
        for pattern in ingredient_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    lines = match.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('•') and '재료' not in line and '📋' not in line:
                            ingredients.append(line)
                break
        
        return ingredients if ingredients else ["재료 정보를 찾을 수 없습니다"]

    def _parse_tips_from_text(self, dish: str, text: str) -> list:
        """자연어 응답에서 조리 팁 추출"""
        import re
        
        tips = []
        
        # 팁 패턴 찾기
        tip_patterns = [
            r"💡\s*조리\s*팁[:\s]*([\s\S]*?)(?=\n\n|$)",
            r"조리\s*팁[:\s]*([\s\S]*?)(?=\n\n|$)",
            r"팁[:\s]*([\s\S]*?)(?=\n\n|$)",
            r"(\d+\.\s*[^\n]+(?:\n\d+\.\s*[^\n]+)*)"
        ]
        
        for pattern in tip_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    lines = match.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and re.match(r'\d+\.', line):
                            # 숫자 제거하고 내용만 추출
                            tip = re.sub(r'^\d+\.\s*', '', line)
                            if tip and '팁' not in tip.lower():
                                tips.append(tip)
                break
        
        return tips if tips else ["조리 팁을 찾을 수 없습니다"]

    async def _handle_selection_if_any(self, message: str) -> dict | None:
        """사용자가 번호/범위로 선택(예: "1번", "2,3번", "1과 3")했을 때, 최근 추천 리스트에서 요리명을 매핑하여 레시피를 반환.
        선택이 없으면 None 반환.
        """
        import re
        text = (message or "").strip()
        if not text:
            return None
        # 숫자 존재 여부 확인
        if not re.search(r"\d", text):
            return None

        # 최근 추천 없으면 종료
        if not getattr(self, "last_suggested_dishes", None):
            return None

        # "1번 2번", "1,3", "1과 3", "2 3 4" 형태 모두 수집
        indices = re.findall(r"\d+", text)
        if not indices:
            return None

        # 중복 제거 및 1-based → 0-based 매핑
        unique_idxs = []
        for s in indices:
            try:
                n = int(s)
                if n >= 1 and n <= len(self.last_suggested_dishes) and n not in unique_idxs:
                    unique_idxs.append(n)
            except Exception:
                continue

        if not unique_idxs:
            return None

        # 선택된 요리명 목록 생성
        chosen_dishes = [self.last_suggested_dishes[i - 1] for i in unique_idxs]

        # 다건 선택이면 첫 번째만 레시피를 반환하고, 나머지는 선택 목록을 안내
        main_dish = chosen_dishes[0]
        recipe = await self.get_recipe_optimized(main_dish)

        # 사용자에게 보여줄 응답 텍스트 구성
        answer_lines = []
        if len(chosen_dishes) > 1:
            answer_lines.append(
                "여러 개를 선택하셨네요. 먼저 1개 레시피부터 안내드릴게요. 나머지 요리도 원하시면 다시 번호를 말씀해주세요."
            )
        answer_lines.append(f"네. {recipe.get('title', main_dish)}의 레시피를 알려드릴게요.")

        self._add_assistant_response("\n".join(answer_lines))
        self.last_intent = "RECIPE"
        return {
            "answer": "\n".join(answer_lines),
            "food_name": recipe.get("title", main_dish),
            "ingredients": recipe.get("ingredients", []),
            "recipe": recipe.get("steps", recipe.get("recipe", []))
        }


# TextAgent 인스턴스 생성
# text_agent = TextAgent()

@tool
async def text_based_cooking_assistant(query: str) -> str:
    """
    텍스트 기반의 요리 관련 질문에 답변할 때 사용합니다.
    예를 들어, 특정 요리의 레시피, 재료, 조리 팁을 물어보거나 음식 종류(한식, 중식 등)를 추천해달라고 할 때 유용합니다.
    유튜브 링크(URL)가 포함된 질문에는 이 도구를 사용하지 마세요.
    사용자의 질문을 그대로 입력값으로 사용하세요.
    """
    # logger.info(f"텍스트 요리 도우미 실행: {query}")
    # result = await text_agent.process_message(query)
    # logger.info(f"------text_service.process_message에서 만들어진 json으로 결과가 나와야 함 )도우미 응답: {result}")
    # return result

    # --- forward_to_text_service의 로직을 그대로 가져와 적용 ---
    logger.info(f"TextAgent 도구 실행: '{query}'에 대한 처리를 위해 {TEXT_SERVICE_URL}/process로 전달합니다.")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"message": query}
            logger.debug("=== 🤍payload for TextAgent Service: %s", payload)

            logger.info(f"=== 🤍TextAgent Service로 요청 전송: {TEXT_SERVICE_URL}/process")
            async with session.post(f"{TEXT_SERVICE_URL}/process", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"TextAgent Service 응답: {result}")
                    # 딕셔너리를 JSON 문자열로 변환하여 반환
                    return json.dumps(result, ensure_ascii=False)
                else:
                    error_text = await response.text()
                    logger.error(f"TextAgent Service 오류 (상태: {response.status}): {error_text}")
                    return {
                        "error": f"TextAgent Service 오류: {response.status}",
                        "message": error_text
                    }
    except aiohttp.ClientConnectorError as e:
        logger.error(f"TextAgent Service 연결 실패: {e}")
        return {
            "error": "TextAgent Service에 연결할 수 없습니다.",
            "message": "8002 서버가 실행 중인지 확인해주세요."
        }
    except Exception as e:
        logger.error(f"TextAgent Service 호출 중 오류: {e}")
        return {
            "error": "TextAgent Service 호출 중 오류가 발생했습니다.",
            "message": str(e)
        }