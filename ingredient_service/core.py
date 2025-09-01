# ingredient_service/core.py
import logging
import json
import base64
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

class IngredientProcessor:
    """재료 검색 및 처리를 위한 내부 처리 클래스"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # 실제 운영에서는 벡터 DB나 상품 DB 연결
        self.mock_products = self._load_mock_products()
    
    def _load_mock_products(self) -> Dict[str, List[Dict]]:
        """목업 상품 데이터 로드"""
        return {
            "계란": [
                {
                    "product_name": "신선한 계란 30구",
                    "price": 8900,
                    "image_url": "https://example.com/egg1.jpg",
                    "product_address": "https://example.com/product/egg1"
                },
                {
                    "product_name": "유기농 계란 15구",
                    "price": 6500,
                    "image_url": "https://example.com/egg2.jpg",
                    "product_address": "https://example.com/product/egg2"
                }
            ],
            "소고기": [
                {
                    "product_name": "한우 등심 200g",
                    "price": 25000,
                    "image_url": "https://example.com/beef1.jpg",
                    "product_address": "https://example.com/product/beef1"
                },
                {
                    "product_name": "양념갈비 500g",
                    "price": 18000,
                    "image_url": "https://example.com/beef2.jpg",
                    "product_address": "https://example.com/product/beef2"
                }
            ],
            "김치": [
                {
                    "product_name": "맛있는 김치 1kg",
                    "price": 12000,
                    "image_url": "https://example.com/kimchi1.jpg",
                    "product_address": "https://example.com/product/kimchi1"
                }
            ],
            "default": [
                {
                    "product_name": "상품을 찾을 수 없습니다",
                    "price": 0,
                    "image_url": "https://example.com/no-image.jpg",
                    "product_address": "https://example.com"
                }
            ]
        }
    
    async def search_by_text(self, query: str) -> Dict[str, Any]:
        """텍스트 기반 재료 검색"""
        try:
            logger.info(f"텍스트 검색 시작: {query}")
            
            # 간단한 키워드 매칭 (실제로는 벡터 검색이나 더 정교한 매칭 사용)
            query_lower = query.lower()
            matched_products = []
            
            for category, products in self.mock_products.items():
                if category in query_lower or any(keyword in query_lower for keyword in self._get_keywords(category)):
                    matched_products.extend(products)
            
            if not matched_products:
                matched_products = self.mock_products["default"]
            
            result = {
                "success": True,
                "data": {
                    "query": query,
                    "results": matched_products[:5],  # 최대 5개 결과
                    "count": len(matched_products)
                },
                "message": f"'{query}'에 대한 검색 결과를 찾았습니다."
            }
            
            logger.info(f"텍스트 검색 완료: {len(matched_products)}개 결과")
            return result
            
        except Exception as e:
            logger.error(f"텍스트 검색 오류: {e}")
            return {
                "success": False,
                "error": f"검색 중 오류가 발생했습니다: {str(e)}",
                "data": {"results": [], "count": 0}
            }
    
    async def search_by_image(self, image_b64: str) -> Dict[str, Any]:
        """이미지 기반 재료 검색"""
        try:
            logger.info("이미지 검색 시작")
            
            # 실제로는 이미지 분석 AI 모델을 사용
            # 여기서는 목업 데이터 반환
            if not image_b64:
                return {
                    "success": False,
                    "error": "이미지 데이터가 없습니다.",
                    "data": {"results": [], "count": 0}
                }
            
            # 이미지 크기 확인 (간단한 검증)
            try:
                image_data = base64.b64decode(image_b64)
                if len(image_data) < 100:  # 너무 작은 이미지
                    return {
                        "success": False,
                        "error": "이미지가 너무 작습니다.",
                        "data": {"results": [], "count": 0}
                    }
            except Exception:
                return {
                    "success": False,
                    "error": "이미지 데이터 형식이 올바르지 않습니다.",
                    "data": {"results": [], "count": 0}
                }
            
            # 목업 결과 반환 (실제로는 이미지 분석 결과)
            mock_results = [
                {
                    "product_name": "이미지에서 인식된 상품",
                    "price": 15000,
                    "image_url": "https://example.com/detected.jpg",
                    "product_address": "https://example.com/product/detected"
                }
            ]
            
            result = {
                "success": True,
                "data": {
                    "results": mock_results,
                    "count": len(mock_results)
                },
                "message": "이미지에서 상품을 인식했습니다."
            }
            
            logger.info("이미지 검색 완료")
            return result
            
        except Exception as e:
            logger.error(f"이미지 검색 오류: {e}")
            return {
                "success": False,
                "error": f"이미지 분석 중 오류가 발생했습니다: {str(e)}",
                "data": {"results": [], "count": 0}
            }
    
    def _get_keywords(self, category: str) -> List[str]:
        """카테고리별 키워드 매핑"""
        keyword_map = {
            "계란": ["달걀", "egg", "에그"],
            "소고기": ["beef", "고기", "육류"],
            "김치": ["kimchi", "발효", "채소"],
        }
        return keyword_map.get(category, [])
