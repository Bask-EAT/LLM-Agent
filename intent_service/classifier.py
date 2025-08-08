
import google.generativeai as genai
import re
import logging
from typing import Dict, List, Optional
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


class IntentClassifier:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.conversation_history = []

    async def classify_intent(self, message: str) -> Dict[str, any]:
        """사용자 입력의 의도를 분류"""
        try:
            # 현재 메시지를 히스토리에 추가
            self.conversation_history.append({"role": "user", "content": message})
            
            # 유튜브 링크 패턴 확인
            if self._is_youtube_link(message):
                intent = "VIDEO"
                confidence = 0.95
                reason = "유튜브 링크가 감지되었습니다."
            else:
                # LLM을 사용한 의도 분류
                intent_result = await self._classify_with_llm(message)
                intent = intent_result["intent"]
                confidence = intent_result["confidence"]
                reason = intent_result["reason"]
            
            result = {
                "intent": intent,
                "confidence": confidence,
                "reason": reason,
                "message": message
            }
            
            logger.info(f"의도 분류 결과: {intent} (신뢰도: {confidence})")
            return result
            
        except Exception as e:
            logger.error(f"의도 분류 중 오류: {e}")
            # 기본값 반환
            return {
                "intent": "TEXT",
                "confidence": 0.5,
                "reason": "분류 중 오류가 발생하여 기본값을 사용합니다.",
                "message": message
            }

    def _is_youtube_link(self, message: str) -> bool:
        """유튜브 링크인지 확인"""
        youtube_patterns = [
            r'youtube\.com/watch\?v=',
            r'youtu\.be/',
            r'youtube\.com/embed/',
            r'youtube\.com/shorts/',
            r'youtube\.com/playlist\?list='
        ]
        
        for pattern in youtube_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        
        return False

    async def _classify_with_llm(self, message: str) -> Dict[str, any]:
        """LLM을 사용한 의도 분류"""
        context = self._get_recent_context(3)
        
        prompt = f"""
        사용자의 메시지를 분석하여 의도를 분류하세요.
        
        대화 컨텍스트:
        {context}
        
        현재 메시지: {message}
        
        분류 기준:
        1. VIDEO: 유튜브 링크나 영상 관련 요청
        2. TEXT: 텍스트 기반 레시피 검색, 요리 질문, 재료 문의 등
        
        다음 JSON 형식으로 출력하세요:
        {{
            "intent": "VIDEO 또는 TEXT",
            "confidence": 0.0-1.0 사이의 신뢰도,
            "reason": "분류 이유"
        }}
        
        예시:
        - "김치찌개 레시피 알려줘" → TEXT
        - "https://youtube.com/watch?v=..." → VIDEO
        - "계란볶음밥 만드는 법" → TEXT
        - "이 영상에서 레시피 추출해줘" → VIDEO
        """
        
        try:
            resp = self.model.generate_content(prompt)
            response_text = self._clean_json_response(resp.text)
            result = self._parse_json_response(response_text)
            
            # 기본값 설정
            result.setdefault("intent", "TEXT")
            result.setdefault("confidence", 0.7)
            result.setdefault("reason", "LLM 분류 결과")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM 분류 오류: {e}")
            return {
                "intent": "TEXT",
                "confidence": 0.6,
                "reason": "LLM 분류 실패로 기본값 사용"
            }

    def _clean_json_response(self, response_text: str) -> str:
        """JSON 응답 정리"""
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.replace("```json", "").replace("```", "")
        return response_text

    def _parse_json_response(self, response_text: str) -> Dict[str, any]:
        """JSON 응답 파싱"""
        import json
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 정보 추출
            return self._extract_info_from_text(response_text)

    def _extract_info_from_text(self, text: str) -> Dict[str, any]:
        """텍스트에서 의도 정보 추출"""
        intent = "TEXT"
        confidence = 0.6
        reason = "텍스트 파싱으로 분류"
        
        # 의도 키워드 확인
        video_keywords = ["영상", "비디오", "youtube", "유튜브", "링크", "추출"]
        text_keywords = ["레시피", "재료", "조리법", "만드는 법", "요리", "팁"]
        
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in video_keywords):
            intent = "VIDEO"
            confidence = 0.8
            reason = "비디오 관련 키워드 감지"
        elif any(keyword in text_lower for keyword in text_keywords):
            intent = "TEXT"
            confidence = 0.8
            reason = "요리 관련 키워드 감지"
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason
        }

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

    def add_assistant_response(self, content: str):
        """어시스턴트 응답을 히스토리에 추가"""
        self.conversation_history.append({"role": "assistant", "content": content})


# IntentClassifier 인스턴스 생성
intent_classifier = IntentClassifier() 