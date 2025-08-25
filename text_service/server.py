from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Literal
import uvicorn
import logging
import json
import sys, os

# 경로 설정을 먼저 수행해 절대 임포트가 가능하도록 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# TextAgent 임포트 (절대 → 실패 시 로컬 패키지 대체)
try:
    from text_service.agent.core import TextAgent
except ModuleNotFoundError:
    from agent.core import TextAgent

from intent_service.planning_agent import run_agent


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TextAgent Server", description="텍스트 기반 레시피 검색 서버")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TextAgent 인스턴스 생성
text_agent = TextAgent()


class TextRequest(BaseModel):
    message: str


class Ingredient(BaseModel):
    item: str
    amount: str
    unit: str


class Product(BaseModel):
    product_name: str
    price: float | int
    image_url: str
    product_address: str


class RecipeModel(BaseModel):
    source: Literal["text", "video", "ingredient_search"]
    food_name: str
    ingredients: List[Union[Ingredient, Product]]
    recipe: List[str]


class ChatResponse(BaseModel):
    chatType: Literal["chat", "cart", "recipe"]
    content: str
    recipes: List[RecipeModel]


@app.post("/process", response_model=ChatResponse)
async def process_message(request: TextRequest):
    """텍스트 기반 레시피 검색 처리"""
    try:
        logger.info(f"=== 💛text_service에서 /process 엔드포인트 호출됨💛 ===")
        logger.info(f"처리할 메시지: {request.message}")

        result = await text_agent.process_message(request.message)
        logger.info(f"TextAgent 처리 결과: {result}")

        # 표준 스키마로 정규화 (content 우선: answer → content로 승격)
        content = str(result.get("content") or result.get("answer") or "").strip()
        food_name = result.get("food_name") or result.get("title") or ""
        raw_ingredients = result.get("ingredients", [])
        steps = result.get("recipe") or result.get("steps") or []

        def to_ingredient(obj):
            if isinstance(obj, dict) and {"item", "amount", "unit"}.issubset(
                obj.keys()
            ):
                return {
                    "item": str(obj.get("item", "")),
                    "amount": str(obj.get("amount", "")),
                    "unit": str(obj.get("unit", "")),
                }
            if isinstance(obj, str):
                return {"item": obj, "amount": "", "unit": ""}
            return {"item": "", "amount": "", "unit": ""}

        normalized_ings: List[dict] = []
        if isinstance(raw_ingredients, list):
            normalized_ings = [to_ingredient(x) for x in raw_ingredients]

        # chatType 결정 로직
        chat_type = "chat"  # 기본값
        if normalized_ings and steps:
            # 실제 레시피 정보가 있는 경우만 recipe로 설정
            if len(normalized_ings) > 0 and len(steps) > 0:
                chat_type = "recipe"
            else:
                chat_type = "chat"
        else:
            # 재료나 레시피가 없는 경우 chat으로 설정
            chat_type = "chat"

        recipe_obj = {
            "source": "text",
            "food_name": food_name,
            "ingredients": normalized_ings,
            "recipe": steps if isinstance(steps, list) else [],
        }

        logger.info(
            f"최종 chatType: {chat_type}, 재료 개수: {len(normalized_ings)}, 레시피 단계 개수: {len(steps)}"
        )

        return ChatResponse(
            chatType=chat_type, content=content or "", recipes=[recipe_obj]
        )
    except Exception as e:
        logger.error(f"ShoppingAgent 처리 오류: {e}")
        raise HTTPException(
            status_code=500, detail="레시피 검색 중 오류가 발생했습니다."
        )


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "service": "TextAgent Server"}


@app.get("/")
async def root():
    """루트 엔드포인트"""
    logger.info("=== 루트 엔드포인트 호출됨 ===")
    return {
        "message": "ShoppingAgent Server is running",
        "endpoints": {
            "/chat": "POST - 채팅 메시지 처리",
            "/process": "POST - 레시피 검색 처리",
            "/health": "GET - 서버 상태 확인",
        },
    }


if __name__ == "__main__":
    logger.info("=== TextAgent Server 시작 ===")
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)
