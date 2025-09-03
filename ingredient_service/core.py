# ingredient_service/core.py
import logging
import json
import base64
from typing import Dict, List, Any, Optional
import os
import httpx
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class IngredientProcessor:
    """재료 검색 및 처리를 위한 내부 처리 클래스"""

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # Embedding API 연결 설정
        self.vector_db_url = os.getenv("VECTOR_DB_API_URL", "http://localhost:8000")
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
                    "product_address": "https://example.com/product/egg1",
                },
                {
                    "product_name": "유기농 계란 15구",
                    "price": 6500,
                    "image_url": "https://example.com/egg2.jpg",
                    "product_address": "https://example.com/product/egg2",
                },
            ],
            "소고기": [
                {
                    "product_name": "한우 등심 200g",
                    "price": 25000,
                    "image_url": "https://example.com/beef1.jpg",
                    "product_address": "https://example.com/product/beef1",
                },
                {
                    "product_name": "양념갈비 500g",
                    "price": 18000,
                    "image_url": "https://example.com/beef2.jpg",
                    "product_address": "https://example.com/product/beef2",
                },
            ],
            "김치": [
                {
                    "product_name": "맛있는 김치 1kg",
                    "price": 12000,
                    "image_url": "https://example.com/kimchi1.jpg",
                    "product_address": "https://example.com/product/kimchi1",
                }
            ],
            "default": [
                {
                    "product_name": "상품을 찾을 수 없습니다",
                    "price": 0,
                    "image_url": "https://example.com/no-image.jpg",
                    "product_address": "https://example.com",
                }
            ],
        }

    async def search_by_text(self, query: str) -> Dict[str, Any]:
        """텍스트 기반 재료 검색 - Embedding API 직접 호출"""
        try:
            logger.info(f"🔍 [DEBUG] 텍스트 검색 시작: '{query}'")
            logger.info(f"🔍 [DEBUG] Vector DB URL: {self.vector_db_url}")

            # Embedding API로 직접 요청
            payload = {"query": query, "top_k": 30}

            logger.info(f"🔍 [DEBUG] 요청 payload: {payload}")
            logger.info(
                f"🔍 [DEBUG] 요청 URL: {self.vector_db_url}/search/crossmodal-text"
            )

            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.info(f"🔍 [DEBUG] HTTP 요청 전송 중...")
                response = await client.post(
                    f"{self.vector_db_url}/search/crossmodal-text",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                logger.info(f"🔍 [DEBUG] 응답 상태 코드: {response.status_code}")
                logger.info(f"🔍 [DEBUG] 응답 헤더: {dict(response.headers)}")
                logger.info(f"🔍 [DEBUG] 응답 원본 텍스트: {response.text}")

                response.raise_for_status()
                search_result = response.json()

                logger.info(f"🔍 [DEBUG] 파싱된 JSON 응답: {search_result}")
                logger.info(f"🔍 [DEBUG] 응답 타입: {type(search_result)}")

            # 결과를 표준 형식으로 변환
            results = search_result.get("results", [])
            logger.info(f"🔍 [DEBUG] results 필드: {results}")
            logger.info(f"🔍 [DEBUG] results 타입: {type(results)}")
            logger.info(
                f"🔍 [DEBUG] results 길이: {len(results) if isinstance(results, list) else 'N/A'}"
            )

            products = []

            for i, item in enumerate(results):
                logger.info(f"🔍 [DEBUG] 결과 {i+1}: {item}")
                logger.info(f"🔍 [DEBUG] 결과 {i+1} 타입: {type(item)}")

                if isinstance(item, dict):
                    product = {
                        "product_name": str(item.get("product_name", "")),
                        "price": item.get("price", 0),
                        "image_url": str(item.get("image_url", "")),
                        "product_address": str(item.get("product_address", "")),
                    }
                    logger.info(f"🔍 [DEBUG] 변환된 상품 {i+1}: {product}")
                    products.append(product)
                else:
                    logger.warning(f"🔍 [DEBUG] 결과 {i+1}이 dict가 아님: {type(item)}")

            result = {
                "success": True,
                "data": {"query": query, "results": products, "count": len(products)},
                "message": f"'{query}'에 대한 검색 결과를 찾았습니다.",
            }

            logger.info(f"🔍 [DEBUG] 최종 결과: {result}")
            logger.info(f"🔍 [DEBUG] 텍스트 검색 완료: {len(products)}개 결과")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Embedding API 호출 오류: {e.response.status_code} - {e.response.text}"
            )
            # API 호출 실패 시 목업 데이터로 폴백
            return await self._fallback_to_mock_data(query, "text")
        except Exception as e:
            logger.error(f"텍스트 검색 오류: {e}")
            # 기타 오류 시 목업 데이터로 폴백
            return await self._fallback_to_mock_data(query, "text")

    async def search_by_image(self, image_b64: str) -> Dict[str, Any]:
        """이미지 기반 재료 검색 - Embedding API 직접 호출"""
        try:
            logger.info("이미지 검색 시작")

            if not image_b64:
                return {
                    "success": False,
                    "error": "이미지 데이터가 없습니다.",
                    "data": {"results": [], "count": 0},
                }

            # 이미지 크기 확인 (간단한 검증)
            try:
                image_data = base64.b64decode(image_b64)
                if len(image_data) < 100:  # 너무 작은 이미지
                    return {
                        "success": False,
                        "error": "이미지가 너무 작습니다.",
                        "data": {"results": [], "count": 0},
                    }
            except Exception:
                return {
                    "success": False,
                    "error": "이미지 데이터 형식이 올바르지 않습니다.",
                    "data": {"results": [], "count": 0},
                }

            # Embedding API로 직접 요청 (multipart/form-data)
            files = {"file": ("image.jpeg", image_data, "image/jpeg")}
            params = {"top_k": 30, "history": "latest"}

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.vector_db_url}/search/image", files=files, params=params
                )
                response.raise_for_status()
                search_result = response.json()

            # 결과를 표준 형식으로 변환
            results = search_result.get("results", [])
            products = []

            for item in results:
                if isinstance(item, dict):
                    product = {
                        "product_name": str(item.get("product_name", "")),
                        "price": item.get("price", 0),
                        "image_url": str(item.get("image_url", "")),
                        "product_address": str(item.get("product_address", "")),
                    }
                    products.append(product)

            result = {
                "success": True,
                "data": {"results": products, "count": len(products)},
                "message": "이미지에서 상품을 인식했습니다.",
            }

            logger.info(f"이미지 검색 완료: {len(products)}개 결과")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Embedding API 호출 오류: {e.response.status_code} - {e.response.text}"
            )
            # API 호출 실패 시 목업 데이터로 폴백
            return await self._fallback_to_mock_data("이미지 검색", "image")
        except Exception as e:
            logger.error(f"이미지 검색 오류: {e}")
            # 기타 오류 시 목업 데이터로 폴백
            return await self._fallback_to_mock_data("이미지 검색", "image")

    async def _fallback_to_mock_data(
        self, query: str, search_type: str
    ) -> Dict[str, Any]:
        """API 호출 실패 시 목업 데이터로 폴백"""
        try:
            logger.info(f"목업 데이터로 폴백: {query} ({search_type})")

            if search_type == "text":
                # 텍스트 검색용 목업 데이터
                query_lower = query.lower()
                matched_products = []

                for category, products in self.mock_products.items():
                    if category in query_lower or any(
                        keyword in query_lower
                        for keyword in self._get_keywords(category)
                    ):
                        matched_products.extend(products)

                if not matched_products:
                    matched_products = self.mock_products["default"]

                result = {
                    "success": True,
                    "data": {
                        "query": query,
                        "results": matched_products[:5],
                        "count": len(matched_products),
                    },
                    "message": f"'{query}'에 대한 검색 결과를 찾았습니다. (목업 데이터)",
                }

            else:  # image
                # 이미지 검색용 목업 데이터
                mock_results = [
                    {
                        "product_name": "이미지에서 인식된 상품",
                        "price": 15000,
                        "image_url": "https://example.com/detected.jpg",
                        "product_address": "https://example.com/product/detected",
                    }
                ]

                result = {
                    "success": True,
                    "data": {"results": mock_results, "count": len(mock_results)},
                    "message": "이미지에서 상품을 인식했습니다. (목업 데이터)",
                }

            return result

        except Exception as e:
            logger.error(f"목업 데이터 폴백 오류: {e}")
            return {
                "success": False,
                "error": f"검색 중 오류가 발생했습니다: {str(e)}",
                "data": {"results": [], "count": 0},
            }

    def _get_keywords(self, category: str) -> List[str]:
        """카테고리별 키워드 매핑"""
        keyword_map = {
            "계란": ["달걀", "egg", "에그"],
            "소고기": ["beef", "고기", "육류"],
            "김치": ["kimchi", "발효", "채소"],
        }
        return keyword_map.get(category, [])
