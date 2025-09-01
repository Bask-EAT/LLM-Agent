import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Gemini API 설정
<<<<<<< HEAD
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
=======
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 의도 분류 최적화 설정
USE_SIMPLE_CLASSIFICATION = os.getenv("USE_SIMPLE_CLASSIFICATION", "true").lower() == "true"
ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
AB_TEST_RATIO = float(os.getenv("AB_TEST_RATIO", "0.5"))

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 성능 모니터링 설정
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
>>>>>>> llm-update

# 애플리케이션 설정
APP_HOST = "0.0.0.0"
APP_PORT = 8000
DEBUG = True

# 쇼핑 관련 설정
EMART_URL = "https://emart.ssg.com/"
CART_URL = "https://pay.ssg.com/cart/dmsShpp.ssg?gnb=cart"

# LangGraph 설정
MAX_ITERATIONS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 1000
