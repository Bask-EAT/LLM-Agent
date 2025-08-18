from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Annotated
import operator

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
    # search_ingredient_by_image,
    # search_ingredient_multimodal
)

# 1. 사용할 도구(Tools) 정의
tools = [
    text_based_cooking_assistant, 
    extract_recipe_from_youtube,
    search_ingredient_by_text,
    # search_ingredient_by_image,
    # search_ingredient_multimodal
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


# 프롬프트 1: 도구 선택 전용
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 요청 의도를 정확히 분석하여, 어떤 도구를 어떻게 호출할지 결정하는 전문가입니다.

    ---
    ### **1단계: 사용자 의도 분석 (`chatType` 결정)**
    가장 먼저 사용자의 메시지를 분석하여 핵심 의도가 '요리 대화'인지 '장바구니 관련'인지 판단합니다.

    - **'요리 대화'로 판단하는 경우:**
      - 메시지에 YouTube URL이 포함되어 있을 때
      - "레시피 알려줘", "만드는 법 알려줘" 등 요리법을 직접 물어볼 때
      - "계란으로 할 수 있는 요리 뭐 있어?" 와 같이 아이디어를 물어볼 때

    - **'장바구니 관련'으로 판단하는 경우:**
      - "계란 찾아줘", "소금 정보 알려줘" 와 같이 상품 정보 자체를 물어볼 때
      - "찾아줘", "얼마야", "가격 알려줘", "정보 알려줘", "구매", "장바구니" 등의 단어가 포함될 때

    ---
    ### **2단계: 의도에 따른 도구 선택 및 호출**
    1단계에서 판단한 의도에 따라 아래 규칙에 맞춰 도구를 호출해야 합니다.

    - '요리 대화' 라면:
      - `extract_recipe_from_youtube` 또는 `text_based_cooking_assistant` 도구를 사용합니다.

    - '장바구니 관련' 이라면:
      - `search_ingredient_by_text` 도구를 사용합니다.

    ---
    ### **도구 호출 세부 규칙 (중요!)**
    - 사용자가 **여러 요리 레시피**를 한 번에 요청했다면(예: "김치찌개랑 된장찌개 레시피"), 반드시 `text_based_cooking_assistant` 도구를 **요리별로 각각** 호출해야 합니다.
    - 사용자가 **번호 선택(예: 1번, 2,3번)** 으로 후속 요청을 했다면, 그 **원문을 그대로** `text_based_cooking_assistant`에 전달해야 합니다.
    """),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])



# 3. 프롬프트(Prompt) 설정 - 에이전트에게 내리는 지시사항
json_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 주어진 도구의 결과(Tool Output)를 분석하여, 정해진 규칙에 따라 최종 JSON으로 완벽하게 변환하는 JSON 포맷팅 전문가입니다.
    당신의 유일한 임무는 JSON을 생성하는 것입니다. **절대로 다른 도구를 호출하지 마세요.**
    대화 기록의 마지막에 있는 ToolMessage의 내용을 바탕으로 JSON을 생성하세요.

    ---
    ### **JSON 생성 규칙:**

    #### **1. `chatType` 결정:**
    - `tool_output`에 'recipe'와 'ingredients'가 포함되어 있으면 `chatType`은 "chat"입니다.
    - `tool_output`에 'product_name'과 'price'가 포함되어 있으면 `chatType`은 "cart"입니다.

    #### **2. 최종 JSON 구조:**

    - **`chatType`이 "chat"일 경우:**
      - `tool_output`의 `answer`를 참고하여 친절한 답변을 생성하세요.
      - `ingredients`는 반드시 **item, amount, unit**을 키로 가지는 객체의 리스트여야 합니다. amount나 unit이 없는 경우(예: '얼음 약간')에는 빈 문자열("")을 값으로 채워주세요.
      - `recipes` 리스트를 `tool_output`의 내용으로 채우세요.
      - 최종 구조는 반드시 아래와 같아야 합니다.
      ```json
      {{
        "chatType": "chat",
        "answer": "요청하신 레시피입니다.",
        "recipes": [
          {{
            "source": "text",
            "food_name": "음식 이름",
            "ingredients": [{{
                "item": "재료명",
                "amount": "양",
                "unit": "단위"
              }},
              {{
                "item": "물",
                "amount": "100",
                "unit": "ml"
              }},
              ...
            ],
            "recipe": ["요리법1", "요리법2", ...]
          }}
        ]
      }}
      ```

    - **`chatType`이 "cart"일 경우:**
      - `tool_output`의 `answer`를 참고하여 친절한 답변을 생성하세요.
      - **[핵심 규칙]** `tool_output`에서 각 상품마다 **`product_name`, `price`, `image_url`, `product_address`** 4개의 키만 정확히 추출하여 `ingredients` 리스트를 만드세요.
      - 최종 구조는 반드시 아래와 같아야 합니다.
      ```json
      {{
        "chatType": "cart",
        "answer": "요청하신 상품을 찾았습니다.",
        "recipes": [
          {{
            "source": "ingredient_search",
            "food_name": "사용자가 검색한 상품명",
            "ingredients": [
                {{
                  "product_name": "상품 이름",
                  "price": 1234,
                  "image_url": "https://...",
                  "product_address": "https://..."
                }}
            ],
            "recipe": []
          }}
        ]
      }}
      ```
    """),
    MessagesPlaceholder(variable_name="messages"), 
])


# # 4. 에이전트 및 실행기 생성
# agent = create_tool_calling_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # 대화 기록을 관리하기 위한 간단한 인메모리 저장소
# chat_history_store = {}


#-----------------------------------------------------------------------------------------------
# create_tool_calling_agent는 LLM이 도구 사용을 '결정'하게 만드는 부분입니다.
# 이 부분은 LangChain Core 기능이므로 변경이 없습니다.
agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)
    
# 1. 에이전트 상태(State) 정의
# 에이전트가 작업하는 동안 유지하고 업데이트할 데이터 구조입니다.
class AgentState(TypedDict):
    # 'messages'는 대화 기록을 담습니다. operator.add는 새 메시지를 리스트에 추가합니다.
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 2. LangGraph의 노드(Node)와 엣지(Edge) 정의
# --- 3개의 전문화된 노드 ---
# 1. 도구 선택 노드
def select_tool(state):
    logger.info("--- [LangGraph] 🧠 Node (select_tool) 실행 ---")
    response = agent.invoke({"input": state["messages"][-1].content, "intermediate_steps": []})
    logger.info(f"--- [LangGraph] 도구 선택 결과: {response} ---")
    return {"messages": response[0].message_log}


# 2. Tool 노드: 미리 만들어진 ToolNode를 사용합니다.
tool_node = ToolNode(tools)


# 3. 최종 답변 생성 노드
def generate_final_answer(state):
    logger.info("--- [LangGraph] ✍️ Node (generate_final_answer) 실행 ---")

    # 'JSON 생성' 역할을 수행하는 체인을 구성합니다.
    chain = json_generation_prompt | llm

    # 수정된 프롬프트에 맞춰, 'messages'라는 키로 전체 대화 기록을 전달합니다.
    final_response = chain.invoke({"messages": state["messages"]})
    
    # 최종 AIMessage를 반환합니다.
    return {"messages": [final_response]}


def should_call_tool(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "action"
    return END


# 4. 그래프(Graph) 생성 및 연결
# 상태 그래프를 만들고 위에서 정의한 노드와 엣지를 연결합니다.
workflow = StateGraph(AgentState)

# 1️⃣ 노드들을 먼저 그래프에 '등록'합니다.
workflow.add_node("agent", select_tool)
workflow.add_node("action", tool_node)
workflow.add_node("formatter", generate_final_answer)

# 2️⃣ 그래프의 시작점을 'agent' 노드로 설정합니다.
workflow.set_entry_point("agent")

# 3️⃣ '등록된' 노드들 사이의 연결선을 정의하는 조건부 엣지를 추가합니다. 'agent' 노드 다음에 should_continue 함수를 실행하여
# 'action'으로 갈지, 'END'로 갈지 결정합니다.
workflow.add_conditional_edges(
    "agent",
    should_call_tool,
    {
        "action": "action",
        END: END,
    },
)

workflow.add_edge("action", "formatter")
workflow.add_edge("formatter", END)

# 4️⃣ 모든 노드와 연결선이 정의된 후, 그래프를 컴파일합니다.
app = workflow.compile()



async def run_agent(user_message: str):
    """사용자 메시지를 받아 에이전트를 실행하고 결과를 반환합니다."""
    logger.info("--- [STEP 0] Agent Start ---")
    
    try:
        # LangGraph 실행
        logger.info("--- [STEP 1] agent_executor.ainvoke 호출 중... ---")
        # result = await agent_executor.ainvoke({
        #     "input": user_message,
        # })
        # inputs = {"messages": [HumanMessage(content=user_message)], "tool_call_count": 0}
        inputs = {"messages": [HumanMessage(content=user_message)]}
        result_state = await app.ainvoke(inputs)
        logger.info("--- [STEP 2] agent_executor.ainvoke가 정상적으로 완료되었습니다. ---")

        # 결과에서 최종 AI 응답 메시지를 추출합니다.
        # output_string = result.get("output", "")
        final_message = result_state["messages"][-1]
        output_string = final_message.content if isinstance(final_message, AIMessage) else ""
        
        # logger.info(f"--- [STEP 3] 출력 문자열 추출 완료. 길이: {len(output_string)}자 ---")
        # logger.debug(f"--- 출력 미리보기: {output_string[:200]}...")  # 앞 200자만 로그에 출력

        # if not output_string or not output_string.strip().startswith(('{', '[')):
        #      logger.error(f"--- [ERROR] 최종 결과가 JSON이 아닙니다: {output_string}")
        #      return json.loads('{"chatType": "error", "answer": "죄송합니다, 답변을 생성하는 데 실패했습니다."}')

         # --- 디버깅 코드 추가 ---
        logger.info("--- [STEP 3] LLM의 원본 응답(Raw Output)을 추출했습니다. ---")
        logger.info(f"\n<<<<<<<<<< RAW OUTPUT START >>>>>>>>>>\n{output_string}\n<<<<<<<<<<< RAW OUTPUT END >>>>>>>>>>>")
        
        if not output_string:
            logger.error("--- [ERROR] LLM의 최종 응답이 비어있습니다.")
            # 비어있는 경우의 에러 처리를 명확하게 합니다.
            return json.loads('{"chatType": "error", "answer": "죄송합니다, 답변을 생성하는 데 실패했습니다. 응답이 비어있습니다."}')



        # 최종 결과에서 ```json ... ``` 부분을 추출
        clean_json_string = ""

        # 1. 먼저 마크다운 블록(```json ... ```)이 있는지 확인하고, 있다면 내부의 JSON만 추출합니다.
        logger.info("--- [STEP 4] 정규식을 사용해 JSON 블록 찾는 중... ---")
        match = re.search(r"```(json)?\s*(\{.*?\})\s*```", output_string, re.DOTALL)
        
        if match:
            clean_json_string = match.group(2).strip()
            logger.info("--- [STEP 5a] 마크다운 블록에서 JSON을 성공적으로 추출했습니다. ---")
        else:
            # 만약 ```json ``` 마크다운을 생성하지 않을 시 전체 문자열 사용 (LLM이 지시를 완전히 따르지 않은 경우일 수 있음)
            logger.warning("--- [STEP 5b] JSON 블록을 찾지 못했습니다. 전체 문자열을 사용합니다. ---")
            clean_json_string = output_string.strip()


        # --- 디버깅 코드 추가 ---
        logger.info("--- [STEP 6] 파싱할 최종 JSON 문자열(Cleaned JSON)을 준비했습니다. ---")
        logger.info(f"\n<<<<<<<<<< CLEAN JSON START >>>>>>>>>>\n{clean_json_string}\n<<<<<<<<<<< CLEAN JSON END >>>>>>>>>>>")
        

        logger.info(f"--- [STEP 7] json.loads()로 문자열을 파싱 시도 중... ---")
        parsed_data = json.loads(clean_json_string)
        
        logger.info(f"--- [STEP 8] json.loads()가 정상적으로 완료되었습니다. 데이터 타입: {type(parsed_data)} ---")
        
        # 마지막 단계: 이 로그가 찍히면, 함수 자체는 성공적으로 끝난 것입니다.
        logger.info("--- ✅ [마지막 단계] 모든 처리가 완료되었습니다. 이제 파싱된 딕셔너리를 반환합니다. ---")
        return parsed_data
    
    

    except Exception as e:
        logger.error(f"--- 🚨 [예외 발생] 예외가 발생했습니다: {e}", exc_info=True)
        raise e