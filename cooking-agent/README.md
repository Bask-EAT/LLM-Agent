# Cooking Agent - 요리 AI 챗봇

요리 전문 AI와 대화하며 레시피, 재료, 조리법을 찾아보세요!

## 🚀 기능

- **의도 분류**: 사용자 입력을 분석하여 적절한 서비스로 라우팅
- **레시피 검색**: 다양한 요리의 레시피와 조리법 제공
- **재료 정보**: 요리에 필요한 재료 목록과 양 제공
- **유튜브 영상 처리**: 유튜브 링크를 통한 레시피 추출
- **실시간 채팅**: AI와의 자연스러운 대화 인터페이스

## 🏗️ 아키텍처

### 프론트엔드 (cooking-agent)
- **Next.js 15** + **React 19** + **TypeScript**
- **Tailwind CSS** + **shadcn/ui** 컴포넌트
- **실시간 채팅 인터페이스**

### 백엔드 서비스들
- **Intent Service (8001)**: 의도 분류 및 유튜브 링크 감지
- **ShoppingAgent Service (8002)**: 요리 관련 처리 (레시피, 재료, 팁)
- **VideoAgent Service (8003)**: 유튜브 영상 처리

## 🛠️ 설치 및 실행

### 1. 백엔드 서비스 실행

```bash
# 프로젝트 루트에서
python run_services.py
```

또는 개별 서비스 실행:

```bash
# Intent Service (8001)
cd intent_service && python server.py

# ShoppingAgent Service (8002)
cd shopping_service && python server.py

# VideoAgent Service (8003)
cd video_service && python server.py
```

### 2. 프론트엔드 실행

```bash
# cooking-agent 폴더에서
cd cooking-agent

# 의존성 설치
npm install
# 또는
pnpm install

# 개발 서버 실행
npm run dev
# 또는
pnpm dev
```

### 3. 접속

브라우저에서 `http://localhost:3000`으로 접속

## 📱 사용법

### 텍스트 기반 요리 질문
- "김치찌개 레시피 알려줘"
- "계란볶음밥 재료가 뭐야?"
- "파스타 조리 팁 알려줘"

### 유튜브 링크 처리
- 유튜브 링크를 붙여넣으면 자동으로 영상에서 레시피 추출

### 서비스 상태 확인
- 왼쪽 사이드바에서 각 서비스의 연결 상태를 실시간으로 확인

## 🔧 개발

### 프로젝트 구조
```
cooking-agent/
├── app/                    # Next.js App Router
│   ├── page.tsx           # 메인 채팅 인터페이스
│   └── layout.tsx         # 레이아웃
├── components/            # UI 컴포넌트
│   └── ui/               # shadcn/ui 컴포넌트
├── lib/                  # 유틸리티
│   └── api.ts            # 백엔드 API 호출 함수
└── package.json          # 프로젝트 설정
```

### API 엔드포인트

#### Intent Service (8001)
- `POST /classify`: 의도 분류
- `GET /health`: 서비스 상태 확인

#### ShoppingAgent Service (8002)
- `POST /chat`: 요리 관련 메시지 처리
- `GET /health`: 서비스 상태 확인

#### VideoAgent Service (8003)
- `POST /process`: 유튜브 영상 처리
- `GET /health`: 서비스 상태 확인

## 🎨 UI 특징

- **반응형 디자인**: 모바일과 데스크톱 모두 지원
- **실시간 상태 표시**: 서비스 연결 상태를 실시간으로 확인
- **동적 재료 목록**: AI 응답에 따라 재료 목록이 자동 업데이트
- **로딩 상태**: AI 응답 생성 중 로딩 애니메이션 표시

## 🔍 문제 해결

### 서비스 연결 실패
1. 모든 백엔드 서비스가 실행 중인지 확인
2. 포트 8001, 8002, 8003이 사용 가능한지 확인
3. 브라우저 개발자 도구에서 네트워크 오류 확인

### CORS 오류
- Next.js rewrite 설정으로 해결됨
- 백엔드 서비스의 CORS 설정 확인

## 📝 라이선스

MIT License 