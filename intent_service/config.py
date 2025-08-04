import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 애플리케이션 설정
APP_HOST = "0.0.0.0"
APP_PORT = 8001
DEBUG = True

# 쇼핑 관련 설정
EMART_URL = "https://emart.ssg.com/"
CART_URL = "https://pay.ssg.com/cart/dmsShpp.ssg?gnb=cart"

# LangGraph 설정
MAX_ITERATIONS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 1000
