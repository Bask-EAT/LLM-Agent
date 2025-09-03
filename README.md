
# LLM-Agent: AI 레시피 어시스턴트 시스템

구글 Gemini API와 LangChain을 활용한 고급 AI 레시피 어시스턴트 시스템입니다. 의도 분류, 텍스트 기반 레시피 검색, 유튜브 영상 레시피 추출, 재료 검색 및 이미지 인식을 통합적으로 제공합니다.

## 🚀 주요 기능

### 🧠 **의도 분류 시스템 (Intent Classification)**
- **Planning Agent**: 사용자 입력을 분석하여 적절한 서비스로 자동 라우팅
- **키워드 기반 분류**: 빠른 응답을 위한 키워드 매칭 시스템
- **LLM 기반 분류**: 정확한 의도 파악을 위한 AI 분석
- **A/B 테스트 지원**: 두 분류 방식의 성능 비교 및 최적화

### 📝 **텍스트 기반 레시피 서비스 (Text Service)**
- **레시피 검색**: 요리명으로 상세한 레시피 조회
- **재료 정보**: 정확한 재료 목록과 양 정보
- **조리 팁**: 요리 관련 노하우와 팁 제공
- **카테고리 추천**: 음식 카테고리별 요리 추천

### 🎥 **비디오 레시피 추출 서비스 (Video Service)**
- **유튜브 링크 분석**: 다양한 유튜브 URL 형식 지원
- **자동 레시피 추출**: 영상 내용에서 레시피 정보 자동 분석
- **구조화된 정보**: 재료, 조리법, 팁을 체계적으로 정리
- **다국어 자막 지원**: 한국어/영어 자막 기반 분석

### 🥕 **재료 검색 서비스 (Ingredient Service)**
- **텍스트 기반 검색**: 재료명으로 상세 정보 조회
- **이미지 인식**: 사진을 통한 재료 자동 인식
- **정규화된 데이터**: 일관된 형식의 재료 정보 제공
- **대체 재료 추천**: 유사한 재료나 대체 가능한 옵션 제시

## 📋 요구사항

- **Python**: 3.8 이상
- **API 키**: Google Gemini API 키
- **메모리**: 최소 4GB RAM 권장
- **저장공간**: 최소 2GB 여유 공간

## 🛠️ 설치 및 실행

### 1. 저장소 클론 및 의존성 설치

```bash
git clone <repository-url>
cd LLM-Agent
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 Gemini API 키를 설정하세요:

```bash
# .env 파일 생성
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. 서버 실행

**메인 서버 실행 (intent-service):**
```bash
cd intent_service
python server.py
```

또는

```bash
cd intent_service
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**개별 서비스 실행 (이전 버전 - 현재는 불필요):**

```bash
# ⚠️ 이전 버전에서는 각 서비스를 개별적으로 실행했지만,
# 현재는 intent-service의 server.py만 실행하면 모든 서비스가 통합되어 작동합니다.

# 텍스트 서비스 (이전 버전)
cd text_service
uvicorn server:app --host 0.0.0.0 --port 8002 --reload

# 비디오 서비스 (이전 버전)
cd video_service
uvicorn server:app --host 0.0.0.0 --port 8003 --reload

# 재료 서비스 (이전 버전)
cd ingredient_service
uvicorn server:app --host 0.0.0.0 --port 8004 --reload
```

### 4. 브라우저에서 접속

```
메인 서버: http://localhost:8001
```

**참고**: 이전 버전에서는 각 서비스를 개별 포트로 접속했지만, 현재는 메인 서버(8001)만 접속하면 모든 기능을 사용할 수 있습니다.

## 🏗️ 프로젝트 구조

```
LLM-Agent/
├── intent_service/           # 🧠 메인 의도 분류 및 통합 서비스
│   ├── server.py            # FastAPI 메인 서버 (포트 8001)
│   ├── planning_agent.py    # LangChain 기반 의도 분류 및 라우팅
│   ├── config.py            # 서비스 설정
│   └── __init__.py
├── text_service/            # 📝 텍스트 기반 레시피 서비스
│   ├── server.py            # 텍스트 서비스 서버 (포트 8002)
│   ├── agent/               # 텍스트 처리 에이전트
│   │   ├── core.py         # 핵심 텍스트 처리 로직
│   │   ├── intent.py       # 의도 분류
│   │   ├── recipes.py      # 레시피 처리
│   │   └── ...
│   ├── config.py            # 텍스트 서비스 설정
│   └── __init__.py
├── video_service/           # 🎥 비디오 레시피 추출 서비스
│   ├── server.py            # 비디오 서비스 서버 (포트 8003)
│   ├── core/                # 비디오 처리 핵심 로직
│   │   ├── extractor.py    # 유튜브 레시피 추출
│   │   └── transcript.py   # 자막 처리
│   ├── config.py            # 비디오 서비스 설정
│   └── __init__.py
├── ingredient_service/      # 🥕 재료 검색 및 이미지 인식 서비스
│   ├── server.py            # 재료 서비스 서버 (포트 8004)
│   ├── core.py              # 재료 처리 핵심 로직
│   ├── tools.py             # 재료 검색 도구
│   ├── schemas.py           # 데이터 스키마
│   └── __init__.py
├── requirements.txt          # Python 의존성 목록
├── README.md                # 프로젝트 문서
└── .env                     # 환경 변수 (사용자 생성)
```

## 🎯 API 엔드포인트

### 메인 서버 (intent-service:8001)

```bash
# 의도 분류 및 통합 처리
POST /process_message
{
  "message": "김치찌개 레시피 알려줘"
}

# 서비스 상태 확인
GET /health

# 통계 정보
GET /stats
```

### 텍스트 서비스 (text-service:8002)

```bash
# 텍스트 기반 레시피 검색
POST /process_message
{
  "message": "계란볶음밥 재료만 알려줘"
}
```

### 비디오 서비스 (video-service:8003)

```bash
# 유튜브 레시피 추출
POST /extract_recipe
{
  "youtube_url": "https://youtube.com/watch?v=..."
}
```

### 재료 서비스 (ingredient-service:8004)

```bash
# 텍스트 기반 재료 검색
POST /search_by_text
{
  "query": "당근"
}

# 이미지 기반 재료 인식
POST /search_by_image
{
  "image": "base64_encoded_image_data"
}
```

## ⚙️ 설정 옵션

### 환경 변수

```bash
# 필수 설정
GEMINI_API_KEY=your_api_key_here

# 선택적 설정
USE_SIMPLE_CLASSIFICATION=true      # 키워드 기반 분류 사용 (기본값: true)
ENABLE_AB_TESTING=false             # A/B 테스트 활성화 (기본값: false)
AB_TEST_RATIO=0.5                  # A/B 테스트 비율 (0.0~1.0)
ENABLE_PERFORMANCE_MONITORING=true # 성능 모니터링 (기본값: true)
```

### 동적 설정 변경

```bash
# 설정 확인
curl http://localhost:8001/stats

# 키워드 기반 분류 비활성화
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"use_simple_classification": false}'

# A/B 테스트 활성화
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"enable_ab_testing": true, "ab_test_ratio": 0.5}'
```

## 🔧 성능 최적화

### 1. 키워드 기반 분류 활성화

```bash
export USE_SIMPLE_CLASSIFICATION=true
```

**성능 향상**: 25% 이상 응답 속도 개선

### 2. A/B 테스트로 성능 비교

```bash
export ENABLE_AB_TESTING=true
export AB_TEST_RATIO=0.5
```

### 3. 성능 모니터링

```bash
curl http://localhost:8001/stats
```

응답 예시:
```json
{
  "status": "success",
  "classification_stats": {
    "simple_classifications": 150,
    "llm_classifications": 50,
    "errors": 2,
    "total": 202,
    "use_simple": true
  },
  "settings": {
    "use_simple_classification": true,
    "enable_ab_testing": false,
    "ab_test_ratio": 0.5
  }
}
```

## 🎨 사용 예시

### 텍스트 기반 요청

```bash
# 레시피 요청
"김치찌개 레시피 알려줘"

# 재료 요청
"계란볶음밥 재료만 알려줘"

# 조리 팁 요청
"된장찌개 조리 팁"

# 추천 요청
"한식 추천해줘"
```

### 유튜브 링크 요청

```bash
# 표준 유튜브 링크
"https://youtube.com/watch?v=dQw4w9WgXcQ"

# 단축 링크
"https://youtu.be/jNQXAC9IVRw"

# 자연어 요청
"이 영상에서 레시피 추출해줘"
```

### 재료 검색

```bash
# 텍스트 검색
"당근"

# 이미지 업로드
[이미지 파일 첨부]
```

## 🔒 보안 고려사항

- **API 키 보안**: 환경 변수를 통한 안전한 API 키 관리
- **CORS 설정**: 적절한 도메인 제한 권장
- **입력 검증**: 사용자 입력에 대한 검증 및 필터링
- **에러 처리**: 민감한 정보 노출 방지

## 🚀 향후 개선 계획

- [ ] **실제 YouTube API 연동**: 더 정확한 영상 정보 추출
- [ ] **다국어 지원**: 영어, 일본어, 중국어 등 추가 언어 지원
- [ ] **음성 입력**: 음성 인식을 통한 자연스러운 상호작용
- [ ] **레시피 저장**: 사용자별 레시피 북마크 및 저장 기능
- [ ] **개인화 추천**: 사용자 선호도 기반 맞춤형 추천
- [ ] **모바일 앱**: React Native 또는 Flutter 기반 모바일 애플리케이션
- [ ] **실시간 협업**: 여러 사용자가 동시에 레시피 편집
- [ ] **영양 정보**: 칼로리, 영양소 등 상세한 영양 정보 제공

## 🐛 문제 해결

### 일반적인 문제

1. **API 키 오류**
   ```bash
   # .env 파일 확인
   cat .env
   
   # 환경 변수 확인
   echo $GEMINI_API_KEY
   ```

2. **포트 충돌**
   ```bash
   # 사용 중인 포트 확인
   netstat -tulpn | grep :8001
   
   # 다른 포트로 실행
   uvicorn server:app --host 0.0.0.0 --port 8005 --reload
   ```

3. **의존성 문제**
   ```bash
   # 가상환경 생성 및 활성화
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # 의존성 재설치
   pip install -r requirements.txt
   ```

## 📞 지원 및 기여

### 이슈 등록
프로젝트에 문제가 발생하거나 개선 제안이 있으시면 GitHub Issues에 등록해주세요.

### 기여하기
1. 프로젝트를 Fork하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [Google Gemini](https://ai.google.dev/) - 강력한 AI 모델 제공
- [LangChain](https://langchain.com/) - AI 애플리케이션 개발 프레임워크
- [FastAPI](https://fastapi.tiangolo.com/) - 현대적이고 빠른 웹 프레임워크
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - 유튜브 다운로드 및 처리

---

**LLM-Agent**로 더 스마트한 요리 경험을 시작하세요! 🍳✨ 
