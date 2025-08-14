import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import random # 모의 데이터를 위해 추가
# 미리 임베딩 모델과 벡터DB 클라이언트를 초기화해둡니다.
# from sentence_transformers import SentenceTransformer
# import pinecone 
# embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
# pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
# index = pinecone.Index("YOUR_INDEX_NAME")

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

# --- 모의(Mock) 벡터 검색 로직 ---
# TODO: 나중에 이 부분을 실제 Vector DB 검색 로직으로 교체합니다.
def mock_vector_search(query: str, search_type: str):
    """실제 벡터 검색을 흉내 내는 함수"""
    logger.info(f"모의 벡터 검색 실행: '{query}' (타입: {search_type})")
    products = [
        {"product_name": f"신선한 {query}", "price": random.randint(5000, 15000), "store": "마켓OO"},
        {"product_name": f"프리미엄 {query} (냉동)", "price": random.randint(10000, 25000), "store": "XX상회"},
        {"product_name": f"간편 손질 {query}", "price": random.randint(7000, 18000), "store": "온라인OO"},
    ]
    return {"query": query, "search_type": search_type, "results": random.sample(products, 2)}

# --- API 엔드포인트 정의 ---

@app.post("/search/text")
async def search_by_text(request: Request):
    """텍스트로 재료 상품을 검색합니다."""
    body = await request.json()
    query_text = body.get("query_text")
    if not query_text:
        raise HTTPException(status_code=400, detail="query_text가 필요합니다.")
    
    # --- 🧡 여기가 실제 벡터 검색 로직으로 교체되는 부분입니다 🧡 ---
    
    # 1. 사용자 쿼리를 벡터로 변환 (임베딩)
    # query_vector = embedding_model.encode(query_text).tolist()

    # 2. 벡터 DB에 유사도 검색 실행
    # search_response = index.query(
    #     vector=query_vector,
    #     top_k=5,  # 5개의 가장 유사한 결과를 가져옵니다.
    #     include_metadata=True
    # )
    
    # 3. 검색 결과를 필요한 JSON 형태로 가공
    # results = []
    # for match in search_response['matches']:
    #     results.append({
    #         "product_name": match['metadata']['product_name'],
    #         "price": match['metadata']['price'],
    #         # ... 기타 필요한 메타데이터
    #     })
    # search_result = {"query": query_text, "results": results}
    # -------------------------------------------------------------
    
    # 지금은 테스트를 위해 여전히 모의 검색을 사용합니다.
    search_result = mock_vector_search(query_text, "text") # 최종적으로 이 줄을 위 로직으로 대체!
    logger.info(f"검색 결과: {search_result}")
    return search_result

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
