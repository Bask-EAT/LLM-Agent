# LLM-Agent 프로젝트

## 🚀 의도 분류 최적화 기능

### 📊 성능 개선 사항

#### 1. 키워드 기반 의도 분류
- **기존**: LLM을 사용한 의도 분류 (느림)
- **개선**: 키워드 매칭 기반 분류 (빠름)
- **성능 향상**: 25% 이상 응답 속도 개선

#### 2. 지원하는 의도 분류
- **RECIPE**: 레시피 요청 ("레시피", "조리법", "만드는 법" 등)
- **INGREDIENTS**: 재료 요청 ("재료", "준비물", "재료만" 등)
- **TIP**: 조리 팁 ("팁", "조리 팁", "꿀팁" 등)
- **CATEGORY**: 추천 요청 ("추천", "추천해줘" 등)
- **SUBSTITUTE**: 재료 대체 ("대신", "대체", "없으면" 등)
- **NECESSITY**: 필요 여부 ("꼭", "필요", "빼도 돼" 등)

### ⚙️ 설정 옵션

#### 환경변수
```bash
# 키워드 기반 분류 사용 (기본값: true)
USE_SIMPLE_CLASSIFICATION=true

# A/B 테스트 활성화 (기본값: false)
ENABLE_AB_TESTING=false

# A/B 테스트 비율 (0.0~1.0, 기본값: 0.5)
AB_TEST_RATIO=0.5

# 성능 모니터링 활성화 (기본값: true)
ENABLE_PERFORMANCE_MONITORING=true
```

#### 동적 설정 변경
```bash
# 설정 확인
curl http://localhost:8002/stats

# 키워드 기반 분류 비활성화
curl -X POST http://localhost:8002/config \
  -H "Content-Type: application/json" \
  -d '{"use_simple_classification": false}'

# A/B 테스트 활성화 (50% 키워드, 50% LLM)
curl -X POST http://localhost:8002/config \
  -H "Content-Type: application/json" \
  -d '{"enable_ab_testing": true, "ab_test_ratio": 0.5}'
```

### 📈 모니터링

#### 통계 확인
```bash
curl http://localhost:8002/stats
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

### 🔧 구현 세부사항

#### 1. 안전한 전환
- 기존 LLM 분류 방식 유지
- 새로운 키워드 분류 방식 추가
- 환경변수로 제어 가능
- 에러 발생 시 안전한 기본값 사용

#### 2. A/B 테스트 지원
- 키워드 vs LLM 분류 성능 비교
- 설정 가능한 테스트 비율
- 실시간 통계 수집

#### 3. 성능 모니터링
- 분류 시간 측정
- 상세한 로깅
- 통계 API 제공

### 🎯 사용 권장사항

#### 1단계: 키워드 기반 분류 활성화
```bash
export USE_SIMPLE_CLASSIFICATION=true
```

#### 2단계: A/B 테스트로 성능 비교
```bash
export ENABLE_AB_TESTING=true
export AB_TEST_RATIO=0.5
```

#### 3단계: 통계 확인 후 최적화
- `/stats` 엔드포인트로 성능 확인
- 필요시 키워드 추가/수정
- 최적 비율 결정

### ⚠️ 주의사항

1. **키워드 정확도**: 모호한 표현은 기본값(RECIPE)으로 분류
2. **다국어 지원**: 한국어/영어 키워드 모두 지원
3. **에러 처리**: 분류 실패 시 안전한 기본값 사용
4. **점진적 적용**: A/B 테스트로 안전한 전환 권장

# AI Recipe Assistant with Intent Classification

구글 Gemini API를 활용한 AI 레시피 어시스턴트입니다. 텍스트 기반 레시피 검색과 유튜브 영상 레시피 추출을 모두 지원합니다.

## 🚀 주요 기능

- **Intent LLM**: 사용자 입력을 분석하여 텍스트 검색과 비디오 분석을 자동 분류
- **텍스트 기반 레시피 검색**: TextAgent를 통한 레시피, 재료, 조리 팁 검색
- **유튜브 영상 레시피 추출**: VideoAgent를 통한 유튜브 링크에서 레시피 자동 추출
- **통합 채팅 인터페이스**: 하나의 채팅창에서 모든 기능 사용 가능
- **실시간 채팅**: WebSocket을 통한 실시간 대화
- **Agent 타입 표시**: 어떤 Agent가 사용되었는지 시각적으로 표시

## 📋 요구사항

- Python 3.8+
- Google Gemini API 키

## 🛠️ 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`env_example.txt`를 참고하여 `.env` 파일을 생성하고 Gemini API 키를 설정하세요:

```bash
cp env_example.txt .env
# .env 파일을 편집하여 GEMINI_API_KEY 설정
```

### 3. 애플리케이션 실행

```bash
python app.py
```

또는

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 브라우저에서 접속

```
http://localhost:8000
```

## 🏗️ 프로젝트 구조

```
dev2/
├── app.py                 # FastAPI 메인 애플리케이션
├── integrated_agent.py    # 통합 에이전트 (Intent 분류 및 Agent 선택)
├── intent_classifier.py   # Intent LLM (사용자 입력 분류)
├── text_service/          # 텍스트 기반 레시피 검색 서비스
├── video_agent.py         # 유튜브 영상 레시피 추출 에이전트
├── config.py             # 설정 관리
├── requirements.txt      # Python 의존성
├── templates/
│   └── index.html       # 메인 HTML 템플릿
├── static/
│   ├── css/
│   │   └── style.css    # CSS 스타일
│   └── js/
│       └── script.js    # JavaScript 로직
└── README.md            # 프로젝트 문서
```

## 🎯 기능 설명

### 1. Intent LLM 분류 시스템

사용자의 입력을 자동으로 분석하여 적절한 Agent로 전달:

- **텍스트 입력**: "김치찌개 레시피 알려줘", "계란볶음밥 재료"
- **유튜브 링크**: "https://youtube.com/watch?v=...", "이 영상에서 레시피 추출해줘"

### 2. TextAgent (텍스트 기반 레시피 검색)

- **레시피 검색**: 요리명으로 레시피 조회
- **재료 검색**: 특정 요리의 재료 목록
- **조리 팁**: 요리 관련 팁과 노하우
- **카테고리 추천**: 음식 카테고리별 요리 추천

### 3. VideoAgent (유튜브 영상 레시피 추출)

- **유튜브 링크 분석**: 다양한 유튜브 링크 형식 지원
- **레시피 추출**: 영상 내용에서 레시피 정보 자동 추출
- **재료 및 조리법**: 구조화된 형태로 정보 제공
- **조리 팁**: 영상에서 추출한 조리 팁

### 4. 지원하는 입력 유형

**텍스트 기반:**
- "김치찌개 레시피 알려줘"
- "계란볶음밥 재료만 알려줘"
- "된장찌개 조리 팁"
- "한식 추천해줘"

**유튜브 링크:**
- "https://youtube.com/watch?v=dQw4w9WgXcQ"
- "https://youtu.be/jNQXAC9IVRw"
- "이 영상에서 레시피 추출해줘"

## 🔧 설정 옵션

`config.py`에서 다음 설정을 변경할 수 있습니다:

- `TEMPERATURE`: AI 응답의 창의성 (0.0 ~ 1.0)
- `MAX_TOKENS`: 최대 토큰 수
- `APP_HOST`, `APP_PORT`: 서버 호스트 및 포트

## 🎨 UI/UX 특징

- **반응형 디자인**: 다양한 화면 크기에 대응
- **부드러운 애니메이션**: 페이지 로드 및 메시지 전송 시 애니메이션
- **모던한 디자인**: 그라데이션과 그림자 효과
- **직관적인 인터페이스**: 사용자 친화적인 채팅 인터페이스

## 🔒 보안 고려사항

- API 키는 환경 변수로 관리
- WebSocket 연결 보안
- XSS 방지를 위한 입력 검증

## 🚀 향후 개선 계획

- [ ] 실제 YouTube API 연동
- [ ] 더 많은 비디오 플랫폼 지원 (네이버TV, 카카오TV 등)
- [ ] 이미지 기반 레시피 인식
- [ ] 음성 입력 지원
- [ ] 레시피 저장 및 북마크 기능
- [ ] 다국어 지원

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 