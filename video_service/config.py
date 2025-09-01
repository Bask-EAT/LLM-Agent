# API 키 등 설정 관리
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# API 키를 변수로 저장
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🌟 이 줄을 추가하여 GEMINI_API_KEY를 LangChain이 찾는 GOOGLE_API_KEY 환경 변수에 직접 할당합니다.
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY