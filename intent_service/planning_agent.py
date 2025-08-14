# planning_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
from dotenv import load_dotenv
import json
import re
import logging


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .env 파일 로드 및 API 키 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 다른 폴더에 있는 모듈을 가져오기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# 부모 디렉토리(프로젝트의 루트 폴더) 경로를 가져옵니다.
project_root = os.path.dirname(current_dir)

# 파이썬이 모듈을 검색하는 경로 리스트에 프로젝트 루트 폴더를 추가합니다.
sys.path.append(project_root)

# 위에서 수정한 파일들로부터 '도구'들을 가져옵니다.
from text_service.agent import text_based_cooking_assistant
from video_service.core.extractor import extract_recipe_from_youtube
from ingredient_service.tools import (
    search_ingredient_by_text,
    search_ingredient_by_image,
    search_ingredient_multimodal
)

# 1. 사용할 도구(Tools) 정의
tools = [
    text_based_cooking_assistant, 
    extract_recipe_from_youtube,
    search_ingredient_by_text,
    search_ingredient_by_image,
    search_ingredient_multimodal
    ]

# 2. LLM 모델 설정
llm = ChatGoogleGenerativeAI(
    # 멀티모달 입력을 처리하려면 Vision 모델 사용을 고려해야 할 수 있습니다.
    # model="gemini-pro-vision",
    model="gemini-2.5-flash", 
    temperature=0, 
    convert_system_message_to_human=True,
    google_api_key=GEMINI_API_KEY,
)

# 3. 프롬프트(Prompt) 설정 - 에이전트에게 내리는 지시사항
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 요청을 3단계에 걸쳐 처리하는 고도로 체계적인 요리 어시스턴트입니다.
     
    ---
    ### **1단계: 사용자 의도 분석 (`chatType` 결정)**
    가장 먼저 사용자의 메시지를 분석하여 핵심 의도가 '요리 대화'인지 '장바구니 관련'인지 판단하고 `chatType`을 결정합니다.

    - **`chatType` = "chat"으로 판단하는 경우:**
      - 메시지에 YouTube URL이 포함되어 있을 때
      - "레시피 알려줘", "만드는 법 알려줘" 등 요리법을 직접 물어볼 때
      - "계란으로 할 수 있는 요리 뭐 있어?" 와 같이 아이디어를 물어볼 때

    - **`chatType` = "cart"으로 판단하는 경우:**
      - "계란 찾아줘", "소금 정보 알려줘" 와 같이 상품 정보 자체를 물어볼 때
      - "장바구니에 담아줘", "구매하고 싶어" 와 같이 명시적인 구매/장바구니 의도가 있을 때
     
    ---
    ### **2단계: 의도에 따른 도구 선택**
    1단계에서 결정한 `chatType`에 따라 사용할 도구를 선택합니다.

    - `chatType`이 **"chat"**이라면:
      - `extract_recipe_from_youtube` 또는 `text_based_cooking_assistant` 도구를 사용해서 레시피 정보를 가져옵니다.
    
    - `chatType`이 **"cart"**이라면:
      - `search_ingredient_by_text` 도구를 사용해서 상품 정보를 검색합니다.
     
    ---
    ### **3단계: 최종 JSON 조립**
    1, 2단계의 결과를 바탕으로, 아래 규칙에 따라 최종 JSON 객체를 **하나만** 생성합니다.

    - **`chatType`이 "chat"일 경우의 JSON 구조:**
      ```json
      {{
        "chatType": "chat",
        "answer": "요청에 대한 친절한 답변 (예: 요청하신 레시피입니다.)",
        "recipes": [
          {{
            "source": "text 또는 video",
            "food_name": "음식 이름",
            "ingredients": ["재료1", "재료2", ...],
            "recipe": ["요리법1", "요리법2", ...]
          }}
        ]
      }}
      ```
     
    - **`chatType`이 "cart"일 경우의 JSON 구조:**
      - `search_ingredient_by_text` 도구로 받은 상품 정보를 **아래 형식으로 변환하여 조립**해야 합니다.
      ```json
      {{
        "chatType": "cart",
        "answer": "요청에 대한 친절한 답변 (예: '계란' 상품을 찾았습니다.)",
        "recipes": [
          {{
            "source": "ingredient_search",
            "food_name": "사용자가 검색한 상품명 (예: 계란)",
            "ingredients": [ 
                {{"product_name": "상품이름A", "price": 10000, ...}},
                {{"product_name": "상품이름B", "price": 12000, ...}}
            ],
            "recipe": []
          }}
        ]
      }}
      ```
    """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# 4. 에이전트 및 실행기 생성
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 대화 기록을 관리하기 위한 간단한 인메모리 저장소
chat_history_store = {}


async def run_agent(user_message: str):
    """사용자 메시지를 받아 에이전트를 실행하고 결과를 반환합니다."""
    logger.info("--- [STEP 0] Agent Start ---")
    
    try:
        logger.info("--- [STEP 1] Calling agent_executor.ainvoke... ---")
        result = await agent_executor.ainvoke({
            "input": user_message,
        })
        logger.info("--- [STEP 2] agent_executor.ainvoke finished successfully. ---")

        output_string = result.get("output", "")
        logger.info(f"--- [STEP 3] Extracted output string. Length: {len(output_string)} chars. ---")
        # 로그가 너무 길어지는 것을 막기 위해 앞 200자만 출력
        logger.debug(f"--- Output preview: {output_string[:200]}...")

        clean_json_string = ""
        logger.info("--- [STEP 4] Attempting to find JSON block using regex... ---")
        match = re.search(r"```json\s*(\{.*?\})\s*```", output_string, re.DOTALL)
        
        if match:
            clean_json_string = match.group(1).strip()
            logger.info("--- [STEP 5a] JSON block found and extracted. ---")
        else:
            logger.warning("--- [STEP 5b] JSON block NOT found. Using the whole string. ---")
            clean_json_string = output_string
        
        logger.info(f"--- [STEP 6] Attempting to parse the string with json.loads()... ---")
        
        # ⚠️ 여기가 가장 유력한 충돌 지점입니다.
        parsed_data = json.loads(clean_json_string)
        
        logger.info(f"--- [STEP 7] json.loads() finished successfully. Data type is: {type(parsed_data)}. ---")
        
        # 마지막 단계: 이 로그가 찍히면, 함수 자체는 성공적으로 끝난 것입니다.
        logger.info("--- ✅ [FINAL STEP] All processing is done. Now returning the parsed dictionary. ---")
        return parsed_data

    except Exception as e:
        # 이 로그가 찍힌다면, 코드에 잡을 수 있는 예외가 발생한 것입니다.
        logger.error(f"--- 🚨 [CAUGHT EXCEPTION] An exception was caught: {e}", exc_info=True)
        raise e