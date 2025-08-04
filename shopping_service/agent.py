import google.generativeai as genai
import os
import json
import logging
from dotenv import load_dotenv

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


class ShoppingAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.conversation_history = []
        self.last_dish = None  # 마지막 언급된 요리명 캐시

    async def process_message(self, message: str) -> dict:
        """메인 메시지 처리 함수"""
        try:
            # 현재 메시지를 히스토리에 추가
            self.conversation_history.append({"role": "user", "content": message})
            
            # 의도 분류 (최적화된 단일 호출)
            intent = await self.classify_intent_optimized(message)
            logger.info(f"의도 분류 결과: {intent}")

            # 의도별 처리
            if intent == "CATEGORY":
                result = await self.recommend_dishes_optimized(message)
                if not result or result == ["추천 요리를 찾을 수 없습니다"]:
                    response_text = "죄송합니다. 추천 요리를 찾을 수 없습니다. 다른 카테고리를 말씀해주세요."
                else:
                    response_text = "다음과 같은 요리들을 추천드려요:\n\n"
                    for i, dish in enumerate(result, 1):
                        response_text += f"{i}. {dish}\n"
                    response_text += "\n어떤 요리의 레시피나 재료를 알고 싶으시면 말씀해주세요!"
                
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
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
                    return {
                        "answer": response_text,
                        "ingredients": [],
                        "recipe": []
                    }
                else:
                    title = result.get("title", dish)
                    ingredients = result.get("ingredients", [])
                    steps = result.get("steps", [])
                    
                    response_text = f"네, 알겠습니다! {title} 레시피에 대한 재료와 조리법입니다.\n\n"
                    response_text += "📋 [재료]\n"
                    for i, ingredient in enumerate(ingredients, 1):
                        response_text += f"{i}. {ingredient}\n"
                    
                    response_text += "\n👨‍🍳 [조리법]\n"
                    for i, step in enumerate(steps, 1):
                        response_text += f"{i}. {step}\n"
                    
                    response_text += f"\n{title} 만드는 데 도움이 되셨나요? 다른 요리도 궁금하시면 언제든 말씀해주세요!"
                    
                    self._add_assistant_response(response_text)
                    return {
                        "answer": response_text,
                        "ingredients": ingredients,
                        "recipe": steps
                    }
                
            elif intent == "INGREDIENTS":
                dish = self._extract_dish_smart(message)
                result = await self.get_ingredients_optimized(dish)
                
                if not result or result == ["재료 정보를 찾을 수 없습니다"]:
                    response_text = f"죄송합니다. {dish}의 재료 정보를 찾을 수 없습니다."
                else:
                    response_text = f"네, 알겠습니다! {dish}에 필요한 재료입니다.\n\n"
                    response_text += "📋 [필요한 재료]\n"
                    for i, ingredient in enumerate(result, 1):
                        response_text += f"{i}. {ingredient}\n"
                    response_text += f"\n{dish} 레시피도 궁금하시면 말씀해주세요!"
                
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
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
                    "ingredients": [],
                    "recipe": result
                }
                
            else:
                response_text = "요리 관련 질문을 해주세요. 레시피, 재료, 조리 팁 등 무엇이든 도와드릴 수 있어요!"
                self._add_assistant_response(response_text)
                return {
                    "answer": response_text,
                    "ingredients": [],
                    "recipe": []
                }
                
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            error_message = "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            return {
                "answer": error_message,
                "ingredients": [],
                "recipe": []
            }

    async def classify_intent_optimized(self, message: str) -> str:
        """최적화된 의도 분류"""
        context = self._get_recent_context(3)  # 최근 3개 메시지만 사용
        
        prompt = f"""
        대화 컨텍스트:
        {context}
        
        현재 메시지: {message}
        
        의도를 분류하세요:
        - CATEGORY: 음식 카테고리 요청 (한식 추천, 중식 추천, 음식추천)
        - RECIPE: 레시피 요청 (레시피 알려줘, 조리법, 만드는 법, 그거 레시피, 그 음식 레시피, 레시피)
        - INGREDIENTS: 재료 요청 (재료 알려줘, 재료만, 그거 재료, 그 음식 재료, 재료)
        - TIP: 조리 팁 (팁 알려줘, 조리 팁, 그거 팁, 팁)
        - OTHER: 그 외

        "그거", "그 음식", "이거" 같은 대명사는 이전 대화의 요리를 참조합니다.
        출력은 CATEGORY, RECIPE, INGREDIENTS, TIP, OTHER 중 하나만
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
            dish = resp.text.strip()
            return dish if dish and len(dish) < 50 else ""
        except Exception as e:
            logger.error(f"LLM 요리명 추출 오류: {e}")
            return ""

    async def recommend_dishes_optimized(self, category: str) -> list:
        """최적화된 요리 추천"""
        prompt = f"""
        "{category}" 요청에 맞는 집에서 할 수 있는 요리 5개를 JSON 배열로 출력하세요.
        예시: ["요리1", "요리2", "요리3", "요리4", "요리5"]
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
            dishes = json.loads(response_text)
            
            if isinstance(dishes, list) and len(dishes) > 0:
                return dishes
            else:
                return ["추천 요리를 찾을 수 없습니다"]
                
        except Exception as e:
            logger.error(f"요리 추천 오류: {e}")
            return ["추천 요리를 찾을 수 없습니다"]

    async def get_recipe_optimized(self, dish: str) -> dict:
        """최적화된 레시피 조회"""
        if self._is_vague_dish(dish):
            return await self._handle_vague_dish_optimized(dish)
        
        prompt = f"""
        '{dish}' 레시피를 JSON으로 작성하세요:
        {{
          "title": "{dish}",
          "ingredients": ["재료1", "재료2"],
          "steps": ["1단계", "2단계"]
        }}
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
            recipe = json.loads(response_text)
            
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
        '{dish}'에 필요한 재료만 JSON 배열로 출력하세요.
        예시: ["재료1", "재료2", "재료3"]
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
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
        '{dish}'를 더 맛있게 만드는 팁 3개를 JSON 배열로 출력하세요.
        예시: ["팁1", "팁2", "팁3"]
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
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
        '{dish}'의 인기 있는 3-5가지 종류를 JSON 배열로 출력하세요.
        예시: ["종류1", "종류2", "종류3"]
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
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

    def _clean_json_response(self, response_text: str) -> str:
        """JSON 응답 정리"""
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.replace("```json", "").replace("```", "")
        return response_text

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

    def _get_recent_context(self, count: int = 3) -> str:
        """최근 대화 컨텍스트 반환"""
        if len(self.conversation_history) == 0:
            return "대화 히스토리가 없습니다."
        
        recent_history = self.conversation_history[-count:]
        context = ""
        for i, msg in enumerate(recent_history):
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            context += f"{i+1}. {role}: {msg['content']}\n"
        
        return context.strip()

    def _add_assistant_response(self, content: str):
        """어시스턴트 응답을 히스토리에 추가"""
        self.conversation_history.append({"role": "assistant", "content": content})


# ShoppingAgent 인스턴스 생성
shopping_agent = ShoppingAgent()
