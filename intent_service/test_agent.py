# test_agent.py

import asyncio
import base64
import json
from planning_agent import run_agent # planning_agent.py 파일에서 run_agent 함수를 가져옵니다.

# --- 설정 ---
# 테스트할 이미지 파일 경로를 지정합니다. 이 파일이 test_agent.py와 같은 위치에 있어야 합니다.
IMAGE_PATH = "사과.jpg" 
USER_PROMPT = "이 이미지랑 비슷한 상품 찾아줘" # 사용자 질문 텍스트

def encode_image_to_base64(filepath: str) -> str:
    """이미지 파일을 읽어 Base64 문자열로 인코딩하는 함수"""
    try:
        with open(filepath, "rb") as image_file:
            # 파일을 바이너리 모드로 읽고, base64로 인코딩한 다음, utf-8 문자열로 디코딩합니다.
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인하세요.")
        return None

async def main():
    """테스트를 실행하는 메인 비동기 함수"""
    
    print(f"--- 🖼️  이미지 파일 인코딩 시작: {IMAGE_PATH} ---")
    image_b64 = encode_image_to_base64(IMAGE_PATH)
    
    if not image_b64:
        return # 이미지 인코딩에 실패하면 테스트를 중단합니다.

    print("--- ✅ 이미지 인코딩 완료 ---")

    # planning_agent의 run_agent 함수가 기대하는 입력 형식(dict)을 만듭니다.
    test_payload = {
        "message": USER_PROMPT,
        "image_b64": image_b64
    }
    
    print("\n--- 🚀 Agent 테스트 시작 ---")
    print(f"--- 🗣️  입력 메시지: '{USER_PROMPT}' ---")

    # 드디어 planning_agent를 실행합니다!
    try:
        result = await run_agent(test_payload)
        
        print("--- ✅ Agent 테스트 성공 ---")
        
        print("\n--- 🎁 최종 결과 (JSON) ---")
        # ensure_ascii=False를 해야 한글이 깨지지 않고 예쁘게 출력됩니다.
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"--- ❌ Agent 테스트 중 오류 발생: {e}")


if __name__ == "__main__":
    # 비동기 main 함수를 실행합니다.
    asyncio.run(main())