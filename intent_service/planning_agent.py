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

# 1. 사용할 도구(Tools) 정의
tools = [text_based_cooking_assistant, extract_recipe_from_youtube]

# 2. LLM 모델 설정 (Planning을 위해서는 고성능 모델을 추천합니다)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    convert_system_message_to_human=True,
    google_api_key=GEMINI_API_KEY,
)

# 3. 프롬프트(Prompt) 설정 - 에이전트에게 내리는 지시사항
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 요청을 분석하여 최적의 도구를 사용해 답변하는 요리 전문 어시스턴트입니다.
     
    ## 최종 답변 생성 규칙
    1.  당신의 최종 목표는 **하나의 JSON 객체**를 생성하는 것입니다.
    2.  이 JSON 객체는 **"answer"**와 **"recipes"** 라는 두 개의 키를 반드시 가져야 합니다.
    3.  **"recipes" 키**: 이 값은 레시피 객체들의 **리스트(list)**여야 합니다.
    4.  **"answer" 키**: 사용자에게 보여줄 친절한 한국어 대답(문자열)을 담습니다.

    ## 매우 중요한 작업 방식
    - 먼저, 사용자의 요청을 처리하는 데 필요한 모든 도구를 호출하여 그 결과들을 수집합니다.
    - **[핵심] 각 도구가 반환한 결과(JSON 객체)를 수정하거나 요약하지 마세요. 받은 그대로를 사용합니다.**
    - 수집한 모든 레시피 객체들을 **단순히 "recipes" 리스트 안에 차례대로 넣어서 조립만 하세요.**
    - 모든 레시피를 리스트에 넣은 후, "answer" 키에 "네, 요청하신 N개의 레시피를 찾았습니다." 와 같은 간단한 요약 멘트를 작성하세요.
    - 이 방식은 당신의 작업 부하를 줄여 안정성을 높이기 위함입니다. '종합'이 아닌 '조립'에 집중해주세요.


    ## 최종 답변 JSON 구조 예시
    ```json
    {{
      "answer": "네, 요청하신 감바스 파스타와 감자튀김 레시피입니다.",
      "recipes": [
        {{
          "food_name": "감바스 파스타",
          "ingredients": ["새우 10마리", ...],
          "recipe": ["1. 마늘을 편으로 썹니다.", ...]
        }},
        {{
          "food_name": "프로 셰프의 완벽 감자튀김",
          "ingredients": ["감자 2개", ...],
          "recipe": ["1. 감자 껍질을 벗깁니다.", ...]
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