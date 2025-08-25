import time

# Service endpoints
TEXT_SERVICE_URL = "http://localhost:8002"

# Cuisine profiles used for recommendations
CUISINE_PROFILES = [
    {"key": "한식", "chef": "강레오, 안성재", "keywords": ["한식", "korean", "코리안"]},
    {"key": "중식", "chef": "Ken Hom", "keywords": ["중식", "중국", "차이니즈"]},
    {"key": "일식", "chef": "Yoshihiro Murata", "keywords": ["일식", "일본", "재패니즈", "japanese", "japan"]},
    {"key": "프랑스식", "chef": "Pierre Koffmann", "keywords": ["프랑스", "프랑스식", "프렌치", "french"]},
    {"key": "이탈리아식", "chef": "Massimo Bottura", "keywords": ["이탈리아", "이탈리아식", "이탈리안", "italian"]},
    {"key": "스페인식", "chef": "José Andrés", "keywords": ["스페인", "스페인식", "spanish"]},
    {"key": "지중해식", "chef": "Yotam Ottolenghi", "keywords": ["지중해", "mediterranean"]},
    {"key": "미국식", "chef": "Gordon Ramsay", "keywords": ["미국", "미국식", "아메리칸", "american"]},
]

# For style detection
STYLE_KEYWORDS = [
    "한식", "중식", "일식", "프랑스", "프랑스식", "이탈리아", "이탈리아식",
    "스페인", "스페인식", "지중해", "미국", "미국식", "korean", "japanese",
    "chinese", "french", "italian", "spanish", "mediterranean", "american"
]

NON_STYLE_HINTS = [
    "재료", "레시피", "만들", "요리", "준비", "굽", "볶", "끓"
]

OTHER_REQUEST_KEYWORDS = [
    "다른 거", "다른것", "다른 요리", "다른 메뉴", "또 추천", "좀 더", "more", "another"
]

# Explicit new-intent trigger phrases
EXPLICIT_NEW_INTENT_KEYWORDS = [
    "새로운", "새 요리", "새 재료", "기존 말고", "그냥 프랑스식", "그냥 이탈리아식",
    "랜덤", "무작위", "다른 걸", "다른 요리", "새 추천"
]

# Default cache TTLs
CACHE_TTL_SECONDS = 300


