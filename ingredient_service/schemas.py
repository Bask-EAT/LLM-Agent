from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    product_name: str
    price: int
    purchase_url: str
    image_url: str

class SearchResult(BaseModel):
    source: str = "ingredient_search" # 이 결과의 출처를 명시
    query: str
    products: List[Product]