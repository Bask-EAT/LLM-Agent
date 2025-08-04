# API 키 등 설정 관리
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# API 키를 변수로 저장
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Google Cloud 인증 파일 경로 설정
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(
    os.path.dirname(__file__), 
    "google-credentials.json"
)

# 환경 변수로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS 