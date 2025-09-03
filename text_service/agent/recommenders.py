import json
import logging
from typing import Dict, List, Optional

from .constants import CUISINE_PROFILES
from .llm import LLMClient

logger = logging.getLogger(__name__)


class Recommenders:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def recommend_by_category(self, message: str, avoid: Optional[List[str]] = None) -> Dict:
        lower_msg = message.lower()
        inferred = next(
            (c for c in CUISINE_PROFILES if any(k in lower_msg or k in message for k in c["keywords"])),
            None,
        )
        if inferred is None:
            logger.info("카테고리 모호: 명시적 국가/스타일이 없어 확인 필요")
            return {"category": "미정", "items": []}
        category_key = inferred["key"]
        chef = inferred["chef"]

        if category_key == "한식":
            avoid_text = ", ".join((avoid or [])[-10:]) if avoid else ""
            prompt = f"""
            당신은 세계 각국의 요리법과 재료에 해박하며, 재료의 유무에 따른 대체재료(특히 한국에서 쉽게 구할 수 있는)까지 파악하고 있는 AI 셰프입니다. 복잡한 과정은 간단하게, 모든 이들이 쉽게 따라 할 수 있도록 명확하고 실용적인 추천을 제공합니다.
            집에서 쉽게 만들 수 있는 가정식/홈스타일 위주로 추천하세요(구하기 쉬운 재료/기본 도구).
            조리 난이도는 쉬움~보통, 준비 시간은 15~40분 내 위주로 구성하세요.
            사용자 메시지: "{message}"
            사용자 메시지에 '다른 거', '다른것', '다른 요리', '다른 메뉴', '또 추천', '좀 더', 'more', 'another'가 포함되면,
            이전 제안과 겹치지 않는 새로운 대안을 우선 추천하세요.
            모든 출력은 한국어로 작성하세요. 요리명은 한국어 표기를 우선 사용하세요
            각 요리별로 간단한 소개를 작성하세요

            한식 요리 5개를 JSON 배열로만 출력하세요(요리명만 출력).
            예시: ["김치찌개", "된장찌개", "불고기", "비빔밥", "잡채"]
            가능하면 아래 목록과 겹치지 않게 다른 요리명을 제안하세요: {avoid_text}
            """
        else:
            avoid_text = ", ".join((avoid or [])[-10:]) if avoid else ""
            prompt = f"""
            당신은 세계 각국의 요리법과 재료에 해박하며, 재료의 유무에 따른 대체재료(특히 한국에서 쉽게 구할 수 있는)까지 파악하고 있는 AI 셰프입니다. 복잡한 과정은 간단하게, 모든 이들이 쉽게 따라 할 수 있도록 명확하고 실용적인 추천을 제공합니다.

            요청 카테고리: {category_key}
            집에서 쉽게 만들 수 있는 홈스타일 버전으로 추천하세요(구하기 쉬운 재료/기본 도구).
            조리 난이도는 쉬움~보통, 준비 시간은 15~40분 내 위주.
            모든 출력은 한국어로 작성하세요. 요리명은 한국어 표기를 우선 사용하세요.
            각 요리별로 간단한 소개를 작성하세요.
            사용자 메시지: "{message}"
            사용자 메시지에 '다른 거', '다른것', '다른 요리', '다른 메뉴', '또 추천', '좀 더', 'more', 'another'가 포함되면,
            이전 제안과 겹치지 않는 새로운 대안을 우선 추천하세요.

            {category_key} 요리 5개를 JSON 배열로만 출력하되, 각 항목은 이름과 한 줄 설명을 포함하세요.
            가능하면 아래 목록과 겹치지 않게 다른 요리명을 제안하세요: {avoid_text}
            출력 형식 예시:
            [
              {{"name": "요리명1", "description": "간단한 한 줄 설명"}},
              {{"name": "요리명2", "description": "간단한 한 줄 설명"}}
            ]
            """

        data = self.llm.generate_json(prompt)
        if data is None:
            return {"category": category_key, "items": []}

        items: List
        if category_key == "한식":
            items = data if isinstance(data, list) else []
            items = [x for x in items if isinstance(x, str) and x.strip()]
        else:
            items = data if isinstance(data, list) else []
            normalized = []
            for it in items:
                if isinstance(it, dict) and it.get("name"):
                    normalized.append({
                        "name": it.get("name", "").strip(),
                        "description": it.get("description", "").strip(),
                    })
                elif isinstance(it, str):
                    normalized.append({"name": it.strip(), "description": ""})
            items = [x for x in normalized if x.get("name")]

        # Post-process: deduplicate using simple normalized name matching
        import re
        def _norm(n: str) -> str:
            s = (n or "").lower().strip()
            s = re.sub(r"[\s·ㆍ・/|()-]+", "", s)
            s = re.sub(r"[^a-z가-힣]", "", s)
            return s
        seen = set()
        deduped: List = []
        for it in items:
            name = it if isinstance(it, str) else it.get("name", "")
            key = _norm(name)
            if not name or key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        items = deduped

        return {"category": category_key, "items": items}


