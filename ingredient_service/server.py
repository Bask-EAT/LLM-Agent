import uvicorn
from fastapi import FastAPI, Request, HTTPException, APIRouter, BackgroundTasks, Body
from typing import List, Union, Literal
from fastapi.middleware.cors import CORSMiddleware
import logging
import httpx
import os
import base64

# --- 설정 (파일 상단에 위치) ---
# 실제 벡터 DB API의 주소를 환경 변수에서 가져옵니다.
# .env 파일에 VECTOR_DB_API_URL="http://실제_벡터DB_주소" 와 같이 설정해야 합니다.
VECTOR_DB_API_URL = os.getenv("VECTOR_DB_API_URL", "http://localhost:8000") # 예시: 기본값 설정
DEFAULT_TOP_K = 10

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ingredient Search Service",
    description="텍스트, 이미지, 멀티모달 벡터 검색을 처리하는 재료 검색 전문가 서버"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search/text")
async def search_by_text(request: Request):
    """
    텍스트로 재료 상품을 검색합니다.
    이 엔드포인트는 planning-agent로부터 요청을 받아,
    실제 벡터 검색 API로 요청을 전달하는 '중개자(Proxy)' 역할을 합니다.
    """
    logger.info("=== 💚 [8004 서버] /search/text 요청 받음 💚 ===")

    try:
        # 1. planning-agent로부터 받은 요청 본문(body)을 파싱합니다.
        incoming_body = await request.json()
        query = incoming_body.get("query")
        if not query:
            logger.error("=== 💚 [8004 서버] 오류: 요청 본문에 'query' 필드가 없습니다.")
            raise HTTPException(status_code=400, detail="'query' 필드가 필요합니다.")
        
        logger.info(f"=== 💚 [8004 서버] 수신된 검색어: '{query}' 💚 ===")

        # 2. 실제 벡터 DB API로 보낼 새로운 요청 본문(payload)을 구성합니다.
        vector_db_payload = {
            "query": query,
            "top_k": incoming_body.get("top_k", DEFAULT_TOP_K) # 요청에 top_k가 있으면 사용, 없으면 기본값
        }
        target_url = f"{VECTOR_DB_API_URL}/search/text"
        
        logger.info(f"=== 💚 [8004 서버] 실제 벡터 DB로 요청 전송 시작. URL: {target_url}, Payload: {vector_db_payload}")

        # 3. httpx를 사용해 실제 벡터 DB API를 비동기적으로 호출합니다.
        async with httpx.AsyncClient() as client:
            response = await client.post(target_url, json=vector_db_payload, timeout=60.0)
            
            logger.info(f"=== 💚 [8004 서버] 실제 벡터 DB로부터 응답 받음. Status: {response.status_code}")
            
            # 응답에 에러가 있으면 예외를 발생시킵니다.
            response.raise_for_status() 
            
            search_result = response.json()
            logger.info(f"=== 💚 [8004 서버] 최종 검색 결과를 planning-agent로 반환합니다.")

            # 표준 스키마로 정규화 (cart 전용)
            query = vector_db_payload.get("query", "")
            items = search_result.get("results", []) if isinstance(search_result, dict) else []
            products: List[dict] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                p = {
                    "product_name": str(it.get("product_name", it.get("name", ""))),
                    "price": it.get("price", 0),
                    "image_url": str(it.get("image_url", "")),
                    "product_address": str(it.get("product_address", "")),
                }
                products.append(p)

            payload = {
                "chatType": "cart",
                "content": f"'{query}' 관련 상품을 찾았습니다.",
                "recipes": [
                    {
                        "source": "ingredient_search",
                        "food_name": str(query),
                        "ingredients": products,
                        "recipe": [],
                    }
                ],
            }
            return payload
        
    except httpx.HTTPStatusError as e:
        # 네트워크 또는 원격 API 에러 처리
        error_message = f"벡터 DB API 호출 중 HTTP 오류 발생: {e.response.status_code} - {e.response.text}"
        logger.error(f"=== ❌ [8004 서버] {error_message}")
        raise HTTPException(status_code=502, detail=error_message) # 502 Bad Gateway
    
    except Exception as e:
        # 기타 예외 처리
        logger.error(f"=== ❌ [8004 서버] 처리 중 알 수 없는 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.")


@app.post("/search/image")
async def search_by_image(
     # ⭐️ body에서 image_b64 문자열을 직접 받도록 수정
    image_b64: str = Body(..., embed=True),
    top_k: int = 10,
    history: str = "latest"
):
    """
    이미지로 유사한 재료 상품을 검색합니다.
    이 엔드포인트는 planning-agent로부터 요청을 받아,
    실제 벡터 검색 API로 요청을 전달하는 '중개자(Proxy)' 역할을 합니다.
    """
    logger.info("=== 💚 [8004 서버] /search/image 요청 받음 💚 ===")

    try:
            # 1. planning-agent로부터 받은 요청 본문(body)을 파싱합니다.
            image_bytes = base64.b64decode(image_b64)
            # image_data = incoming_body.get("image_data") # Base64 인코딩된 이미지 데이터
            
            if not image_bytes:
                logger.error("=== 💚 [8004 서버] 오류: 요청 본문에 'image_bytes' 필드가 없습니다.")
                raise HTTPException(status_code=400, detail="'image_bytes' 필드가 필요합니다.")
                
            logger.info(f"=== 💚 [8004 서버] Base64 이미지 데이터 디코딩 완료 (길이: {len(image_bytes)}) 💚 ===")

            # 2. ⭐️ 8000 서버로 전달할 form-data 생성
            files = {'file': ('image.jpeg', image_bytes, 'image/jpeg')}
            params = {'top_k': top_k, 'history': history}
                # 실제 벡터 DB의 이미지 검색 URL을 지정합니다. (예시: /search/image)
            target_url = f"{VECTOR_DB_API_URL}/search/image"
            
            logger.info(f"=== 💚 [8004 서버] 실제 벡터 DB로 요청 전송 시작. URL: {target_url}")

            # 3. httpx를 사용해 실제 벡터 DB API를 비동기적으로 호출합니다.
            async with httpx.AsyncClient() as client:
                response = await client.post(target_url, files=files, params=params, timeout=30.0)
                
                logger.info(f"=== 💚 [8004 서버] 실제 벡터 DB로부터 응답 받음. Status: {response.status_code}")
                
                response.raise_for_status()     # HTTP 오류 발생 시 예외 발생
                
                search_result = response.json()
                logger.info(f"=== 💚 [8004 서버] 최종 검색 결과: {search_result}")
                logger.info("=== 💚 [8004 서버] 최종 검색 결과를 planning-agent로 반환합니다.")
                return response.json()

                # 4. planning-agent가 이해할 수 있는 표준 스키마로 정규화합니다.
                # 텍스트 검색과 동일한 'cart' 포맷을 사용합니다.
                # items = search_result.get("results", []) if isinstance(search_result, dict) else []
                # products: List[dict] = []
                # for it in items:
                #     if not isinstance(it, dict):
                #         continue
                #     p = {
                #         "product_name": str(it.get("product_name", it.get("name", ""))),
                #         "price": it.get("price", 0),
                #         "image_url": str(it.get("image_url", "")),
                #         "product_address": str(it.get("product_address", "")),
                #     }
                #     products.append(p)

                # payload = {
                #     "chatType": "cart",
                #     "content": "이미지와 관련된 상품을 찾았습니다.",
                #     "recipes": [
                #         {
                #             "source": "ingredient_search",
                #             "food_name": "이미지 기반 검색", # 이미지 검색이므로 특정 음식 이름 대신 일반적인 텍스트 사용
                #             "ingredients": products,
                #             "recipe": [],
                #         }
                #     ],
                # }
                # return payload
                
    except httpx.HTTPStatusError as e:
            error_message = f"벡터 DB API 호출 중 HTTP 오류 발생: {e.response.status_code} - {e.response.text}"
            logger.error(f"=== ❌ [8004 서버] {error_message}")
            raise HTTPException(status_code=502, detail=error_message)
        
    except Exception as e:
            logger.error(f"=== ❌ [8004 서버] 처리 중 알 수 없는 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.")


@app.post("/search/multimodal")
async def search_by_multimodal(request: Request):
    """이미지와 텍스트를 함께 사용하여 재료 상품을 검색합니다."""
    body = await request.json()
    query_text = body.get("query_text")
    image_data = body.get("image_data")
    if not query_text or not image_data:
        raise HTTPException(status_code=400, detail="query_text와 image_data가 모두 필요합니다.")

    # 데모: 표준 스키마 빈 카트 응답
    return {
        "chatType": "cart",
        "content": "멀티모달 검색은 아직 지원되지 않습니다.",
        "recipes": [
            {
                "source": "ingredient_search",
                "food_name": query_text or "",
                "ingredients": [],
                "recipe": [],
            }
        ],
    }

if __name__ == "__main__":
    # 다른 서비스와 겹치지 않는 새 포트(8004)를 사용합니다.
    uvicorn.run(app, host="0.0.0.0", port=8004)
