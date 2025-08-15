import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import httpx
import os

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
            response = await client.post(target_url, json=vector_db_payload, timeout=30.0)
            
            logger.info(f"=== 💚 [8004 서버] 실제 벡터 DB로부터 응답 받음. Status: {response.status_code}")
            
            # 응답에 에러가 있으면 예외를 발생시킵니다.
            response.raise_for_status() 
            
            search_result = response.json()
            logger.info(f"=== 💚 [8004 서버] 최종 검색 결과를 planning-agent로 반환합니다.")
            
            return search_result
        
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
async def search_by_image(request: Request):
    """이미지로 유사한 재료 상품을 검색합니다."""
    body = await request.json()
    image_data = body.get("image_data") # Base64 인코딩된 이미지 데이터라고 가정
    if not image_data:
        raise HTTPException(status_code=400, detail="image_data가 필요합니다.")

    # 실제로는 이 부분에서 이미지를 임베딩하여 벡터 검색을 수행합니다.
    search_result = mock_vector_search("이미지 속 재료", "image")
    return search_result

@app.post("/search/multimodal")
async def search_by_multimodal(request: Request):
    """이미지와 텍스트를 함께 사용하여 재료 상품을 검색합니다."""
    body = await request.json()
    query_text = body.get("query_text")
    image_data = body.get("image_data")
    if not query_text or not image_data:
        raise HTTPException(status_code=400, detail="query_text와 image_data가 모두 필요합니다.")

    # 실제로는 이 부분에서 이미지와 텍스트를 함께 임베딩하여 벡터 검색을 수행합니다.
    search_result = mock_vector_search(f"'{query_text}'와 비슷한 이미지 속 재료", "multimodal")
    return search_result

if __name__ == "__main__":
    # 다른 서비스와 겹치지 않는 새 포트(8004)를 사용합니다.
    uvicorn.run(app, host="0.0.0.0", port=8004)
