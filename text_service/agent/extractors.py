import re
from typing import List


PRONOUNS = ["그거", "그 음식", "이거", "저거", "그것", "이것"]


def find_dish_by_pattern(message: str) -> str:
    # 괄호 내 영문/보조 표기 제거 후 공백 정리
    text = re.sub(r"\([^)]*\)", "", message or "").strip()
    
    # 다단어 + 키워드(레시피/만드는 법/조리법/팁/재료) 앞 구절 캡처
    patterns = [
        r"([가-힣A-Za-z\s]{2,30})\s+(?:레시피|만드는\s*법|조리법|팁)",
        r"([가-힣A-Za-z\s]{2,30})\s+재료",
        # 단어 1개 패턴(이전 호환)
        r"([가-힣]+)\s+레시피",
        r"([가-힣]+)\s+만드는\s+법",
        r"([가-힣]+)\s+조리법",
        r"([가-힣]+)\s+팁",
        r"([가-힣]+)\s+재료",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            name = re.sub(r"\s+", " ", name).strip()
            if 1 < len(name) < 30:
                return name
    return ""


def normalize_ingredient_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    name = text
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[0-9]+\s*[gGkKmMlL컵tspTB]+", "", name)
    name = re.sub(r"[0-9.,/%]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def tokenize_korean_phrase(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[가-힣A-Za-z]+", text)
    return [t for t in tokens if len(t) >= 2]


def match_ingredient_from_inventory(message: str, inventory: List[str]) -> str:
    if not inventory:
        return ""
    message_text = message or ""
    message_tokens = set(tokenize_korean_phrase(message_text))
    if not message_tokens:
        return ""
    best_match = (0, "")
    for ing in inventory:
        base = normalize_ingredient_name(ing)
        if not base:
            continue
        base_tokens = tokenize_korean_phrase(base)
        if not base_tokens:
            continue
        matched_tokens = [t for t in base_tokens if t in message_tokens or t in message_text]
        if not matched_tokens:
            if base and base in message_text:
                score = len(base)
            else:
                score = 0
        else:
            score = sum(len(t) for t in matched_tokens)
        if score > best_match[0]:
            best_match = (score, ing)
    return best_match[1]


def map_to_inventory(name: str, inventory: List[str]) -> str:
    if not name or not inventory:
        return ""
    name_norm = normalize_ingredient_name(name)
    message_like = name_norm
    best = (0, "")
    for ing in inventory:
        base = normalize_ingredient_name(ing)
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


