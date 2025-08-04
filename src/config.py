# API 키 등 설정 관리
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# API 키를 변수로 저장
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")