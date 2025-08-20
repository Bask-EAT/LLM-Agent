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
    search_ingredient_by_image,
    # search_ingredient_multimodal
)

# 1. 사용할 도구(Tools) 정의
tools = [
    text_based_cooking_assistant, 
    extract_recipe_from_youtube,
    search_ingredient_by_text,
    search_ingredient_by_image,
    # search_ingredient_multimodal
    ]

# 2. LLM 모델 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    convert_system_message_to_human=True,
    google_api_key=GEMINI_API_KEY,
    timeout=60,  # <-- ⭐️ 이 부분을 추가해 주세요! (단위: 초)
)


# 프롬프트 1: 도구 선택 전용 - 채팅 히스토리 컨텍스트 지원
# tool_calling_prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     당신은 사용자의 요청 의도를 정확히 분석하여, 어떤 도구를 어떻게 호출할지 결정하는 전문가입니다.
#     대화 히스토리가 제공되는 경우, 이전 컨텍스트를 고려하여 사용자의 의도를 더 정확히 파악하세요.

#     ---
#     ### **1단계: 사용자 의도 분석 (`chatType` 결정)**
     
#     - 메시지가 텍스트이면 기존 규칙 적용
#     - 메시지가 이미지이면:
#       - 장바구니 관련이라면: `search_ingredient_by_image` 호출
#       - 요리 재료 확인 목적이면: `search_ingredient_by_image` 호출
#       - 반드시 이미지 URL이나 base64 형태의 입력 정보를 도구에 전달
    
#     **대화 히스토리가 있는 경우:**
#     - 이전 대화 맥락을 반드시 고려하세요
#     - 사용자가 번호나 짧은 응답("4번", "첫 번째")을 했다면, 이전 AI가 제시한 선택지와 연결하여 이해하세요
#     - 이전에 특정 요리나 상품에 대해 논의했다면, 해당 컨텍스트를 유지하세요
    
#     **의도 분류:**
#     - **'요리 대화'로 판단하는 경우:**
#       - 메시지에 YouTube URL이 포함되어 있을 때
#       - "레시피 알려줘", "만드는 법 알려줘" 등 요리법을 직접 물어볼 때
#       - "계란으로 할 수 있는 요리 뭐 있어?" 와 같이 아이디어를 물어볼 때
#       - **이전에 요리 관련 선택지를 제시했고, 사용자가 번호나 선택을 한 경우**

#     - **'장바구니 관련'으로 판단하는 경우:**
#       - "계란 찾아줘", "소금 정보 알려줘" 와 같이 상품 정보 자체를 물어볼 때
#       - "찾아줘", "얼마야", "가격 알려줘", "정보 알려줘", "구매", "장바구니" 등의 단어가 포함될 때

#     ---
#     ### **2단계: 의도에 따른 도구 선택 및 호출**
    
#     - **'요리 대화' 라면:**
#       - `extract_recipe_from_youtube` 또는 `text_based_cooking_assistant` 도구를 사용합니다.
#       - **컨텍스트가 있는 경우, 전체 대화 맥락을 포함하여 도구에 전달하세요**
#       - 이미지를 통한 재료 확인이면 `search_ingredient_by_image` 사용

#     - **'장바구니 관련' 이라면:**
#       - 텍스트 상품 검색: `search_ingredient_by_text`
#       - 이미지 상품 검색: `search_ingredient_by_image`

#     ---
#     ### **도구 호출 세부 규칙 (중요!)**
#     - 사용자가 **여러 요리 레시피**를 한 번에 요청했다면(예: "김치찌개랑 된장찌개 레시피"), 반드시 `text_based_cooking_assistant` 도구를 **요리별로 각각** 호출해야 합니다.
#     - **채팅 히스토리가 있고 사용자가 번호 선택(예: 1번, 2,3번, 4번) 으로 후속 요청을 했다면:**
#       - 이전 대화 맥락을 포함한 전체 대화를 `text_based_cooking_assistant`에 전달해야 합니다
#       - 단순히 "4번"만 전달하지 말고, "이전에 볶음밥 레시피 선택지를 제시했고 사용자가 4번(파인애플 볶음밥)을 선택했음"과 같은 맥락 정보를 포함하여 전달하세요
#     """),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])
# -------------------------------------------------------------------------------
# 프롬프트 1: 도구 선택 전용 -> 극도로 단순화된 한국어 버전
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 사용자의 요청을 올바른 도구로 연결하는 라우팅 전문가입니다. 당신의 유일한 임무는 사용자의 입력을 바탕으로 올바른 도구를 선택하는 것입니다. 질문에 직접 답변하지 마세요.

- 사용자가 이미지를 제공하면, 반드시 `search_ingredient_by_image` 도구를 사용해야 합니다.
- 사용자가 YouTube URL을 제공하면, 반드시 `extract_recipe_from_youtube` 도구를 사용해야 합니다.
- 사용자가 텍스트만으로 상품을 찾아달라고 요청하면, 반드시 `search_ingredient_by_text` 도구를 사용해야 합니다.
- 요리나 레시피에 대한 그 외 모든 질문에는, 반드시 `text_based_cooking_assistant` 도구를 사용해야 합니다.
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
     
    #### 이미지 검색 결과 처리:
    - `search_ingredient_by_image` 호출 결과는 `cart` 형식 JSON으로 변환
    - 각 결과는 반드시 `product_name`, `price`, `image_url`, `product_address` 포함
    - 텍스트 입력과 혼합된 경우에도 동일한 규칙 적용

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
# 1. 도구 선택 노드 - 채팅 히스토리 컨텍스트 지원
async def select_tool(state):
    logger.info("--- [LangGraph] 🧠 Node (select_tool) 실행 ---")
    
    # 전체 대화 히스토리를 컨텍스트로 활용
    messages = state["messages"]
    # ---------------아래내용 일단 주석처리하고 버그 확인용 코드 추가함------------------
    # if len(messages) > 1:
    #     # 여러 메시지가 있는 경우, 대화 히스토리 컨텍스트 구성
    #     context_parts = []
    #     for i, msg in enumerate(messages):
    #         if isinstance(msg, HumanMessage):
    #             context_parts.append(f"사용자: {msg.content}")
    #         elif isinstance(msg, AIMessage):
    #             context_parts.append(f"AI: {msg.content}")
        
    #     # 전체 대화 맥락과 최신 요청을 결합
    #     full_context = "\n".join(context_parts[:-1])  # 마지막 메시지 제외한 이전 맥락
    #     latest_request = messages[-1].content
        
    #     input_text = f"이전 대화 맥락:\n{full_context}\n\n최신 사용자 요청: {latest_request}"
    #     logger.info(f"--- [LangGraph] 대화 히스토리 컨텍스트 포함 ({len(messages)}개 메시지) ---")
    # else:
    #     # 단일 메시지인 경우 기존 방식 사용
    #     input_text = messages[-1].content
    #     logger.info("--- [LangGraph] 단일 메시지 처리 ---")
    
    # response = await agent.ainvoke({"input": input_text, "intermediate_steps": []})
    # logger.info(f"--- [LangGraph] 도구 선택 결과: {response} ---")
    # return {"messages": response[0].message_log}
    # ------------------------------------------------------------
    # ⭐️⭐️⭐️ 중요 버그 수정 ⭐️⭐️⭐️
    # 이전 코드에서는 이미지/텍스트 복합 메시지(HumanMessage)를 텍스트로만 변환하면서
    # 이미지 데이터가 유실되었습니다. HumanMessage 객체 자체를 넘겨야 합니다.
    # 따라서 복잡한 input_text 생성 로직 대신, 마지막 메시지를 그대로 사용합니다.
    last_message = messages[-1]

    logger.info(f">>> 지금부터 LLM (agent.ainvoke)을 호출합니다. 여기서 멈추는지 확인하세요... Input Type: {type(last_message)}")
    
    # 수정된 부분: input_text 대신 last_message 객체를 input으로 전달
    response = await agent.ainvoke({"input": last_message, "intermediate_steps": []})
    logger.info(f"LLM 원본 응답: {response.content}")
    
    logger.info(">>> LLM 호출이 성공적으로 완료되었습니다!") # 이 로그가 보인다면 멈춤 현상 해결!

    # response는 AgentAction 또는 AgentFinish 객체를 포함할 수 있습니다.
    # LangGraph 상태에 맞게 AIMessage로 변환하여 추가해야 합니다.
    if isinstance(response, AgentFinish):
        # 도구를 호출하지 않고 끝나는 경우, 결과를 AIMessage로 변환
        logger.info(f"--- [LangGraph] Agent가 도구 호출 없이 종료. 결과: {response.return_values}")
        return {"messages": [AIMessage(content=response.return_values.get('output', ''))]}
        # return {"messages": [AIMessage(content=response.return_values['output'])]}
    
    # 도구를 호출하는 경우 (AgentAction)
    # create_tool_calling_agent는 tool_calls를 포함한 AIMessage를 생성해줍니다.
    # 하지만 여기서는 직접 AIMessage를 구성해야 할 수 있습니다.
    # response 객체 구조를 보고 AIMessage를 생성합니다.
    # AgentExecutor가 해주던 역할을 직접 구현하는 단계입니다.
    # AgentAction 객체를 AIMessage의 tool_calls 형식으로 변환합니다.
    # AgentAction은 단일 도구 호출을 나타냅니다.
    logger.info(f"--- [LangGraph] Agent가 도구 호출 결정: {response.tool}")
    
    # create_tool_calling_agent의 결과는 이미 AIMessage 형태일 수 있습니다.
    # LangChain 버전에 따라 다를 수 있으므로 response의 타입을 확인하는 것이 좋습니다.
    # 여기서는 response가 tool_calls를 가진 AIMessage라고 가정합니다.
    
    # agent.invoke의 결과가 AgentAction/AgentFinish 이므로 AIMessage로 변환이 필요
    # tool_calls = []
    # if isinstance(response, AgentAction):
    #      tool_calls.append({
    #          "name": response.tool,
    #          "args": response.tool_input,
    #          "id": response.log.split('tool_call_')[-1].strip() if 'tool_call_' in response.log else "call_1234"
    #      })
    tool_calls = [{
        "name": response.tool,
        "args": response.tool_input,
        "id": "call_" + os.urandom(4).hex() # 임의의 호출 ID 생성
    }]

    ai_message_with_tool_calls = AIMessage(content="", tool_calls=tool_calls)
    
    logger.info(f"--- [LangGraph] 도구 선택 결과: {ai_message_with_tool_calls} ---")
    return {"messages": [ai_message_with_tool_calls]}


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


# 채팅 히스토리를 LangGraph 메시지 형식으로 변환하는 함수
def convert_chat_history_to_messages(chat_history: list) -> list:
    """
    프론트엔드에서 받은 채팅 히스토리를 LangGraph 메시지 형식으로 변환합니다.
    
    [처리 가능한 형식 1: 텍스트 메시지]
    [
        {"role": "user", "content": "라멘 레시피 알려줘"},
        {"role": "assistant", "content": "요청하신 레시피입니다."},
        {"role": "user", "content": "볶음밥 레시피 알려줘"},
        {"role": "assistant", "content": "어떤 볶음밥 레시피를 원하시나요?\n\n1. 김치 볶음밥\n2. 새우 볶음밥\n3. 게살 볶음밥\n4. 파인애플 볶음밥\n\n다른 원하시는 볶음밥 종류가 있으시면 말씀해주세요!"},
        {"role": "user", "content": "4번"}
    ]

    [처리 가능한 형식 2: 이미지 + 텍스트 멀티모달 메시지]
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "이 이미지 분석해서 재료 찾아줘"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }
    """
    messages = []
    
    for msg in chat_history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if not content:
            continue
            
        if role == "user":
            messages.append(HumanMessage(content=content))    # content가 문자열이든 리스트든 HumanMessage가 알아서 처리합니다.
        elif role == "assistant":
            messages.append(AIMessage(content=content))   # AI 메시지는 항상 텍스트이므로 그대로 사용합니다.
        else:
            # 알 수 없는 role은 user로 처리
            logger.warning(f"알 수 없는 role '{role}', user로 처리합니다.")
            messages.append(HumanMessage(content=content))
    
    logger.info(f"채팅 히스토리를 {len(messages)}개의 LangGraph 메시지로 변환했습니다.")
    return messages


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



async def run_agent(input_data: dict):
    """
    사용자 입력을 받아 에이전트를 실행하고 결과를 반환합니다.
    input_data: 
        - 단일 메시지: {"message": str, "image_b64": Optional[str]}
        - 채팅 히스토리: {"chat_history": list} 
    """
    logger.info("--- [STEP 0] Agent Start ---")
    
    try:
        # 입력 데이터 처리: 채팅 히스토리 또는 단일 메시지
        if "chat_history" in input_data:
            # 채팅 히스토리 처리 (기존 로직 유지, 단, 히스토리 내 이미지 포맷은 위 convert 함수 참고)
            chat_history = input_data["chat_history"]
            logger.info(f"--- [STEP 1a] 채팅 히스토리 처리: {len(chat_history)}개 메시지 ---")
            messages = convert_chat_history_to_messages(chat_history)
            
            # 최신 사용자 메시지가 있는지 확인
            if not messages or not isinstance(messages[-1], HumanMessage):
                logger.warning("채팅 히스토리의 마지막 메시지가 사용자 메시지가 아닙니다.")
                # 빈 사용자 메시지 추가
                messages.append(HumanMessage(content=""))
        else:
            # 단일 메시지 처리 (기존 방식)
            user_message = input_data.get("message", "")
            image_b64 = input_data.get("image_b64")
            logger.info(f"--- [STEP 1b] 단일 메시지 처리: {user_message} ---")
            
            if image_b64:
                logger.info("--- [STEP 1c] 이미지 데이터가 포함되었습니다. ---")
                # 이미지가 있는 경우, 텍스트와 이미지를 함께 포함하는 content 리스트 생성
                content = [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"
                    }
                ]
                messages = [HumanMessage(content=content)]
            else:
                # 텍스트만 있는 경우
                messages = [HumanMessage(content=user_message)]
          
        if not messages:
             raise ValueError("처리할 메시지가 없습니다.")
        
        # LangGraph 실행
        logger.info("--- [STEP 2] app.ainvoke 호출 중... ---")
        inputs = {"messages": messages}
        result_state = await app.ainvoke(inputs)
        logger.info("--- [STEP 3] app.ainvoke가 정상적으로 완료되었습니다. ---")

        # 결과에서 최종 AI 응답 메시지를 추출합니다.
        # output_string = result.get("output", "")
        final_message = result_state["messages"][-1]
        output_string = final_message.content if isinstance(final_message, AIMessage) else ""
        
        # logger.info(f"--- [STEP 4] 출력 문자열 추출 완료. 길이: {len(output_string)}자 ---")
        # logger.debug(f"--- 출력 미리보기: {output_string[:200]}...")  # 앞 200자만 로그에 출력

        # if not output_string or not output_string.strip().startswith(('{', '[')):
        #      logger.error(f"--- [ERROR] 최종 결과가 JSON이 아닙니다: {output_string}")
        #      return json.loads('{"chatType": "error", "answer": "죄송합니다, 답변을 생성하는 데 실패했습니다."}')

         # --- 디버깅 코드 추가 ---
        logger.info("--- [STEP 4] LLM의 원본 응답(Raw Output)을 추출했습니다. ---")
        logger.info(f"\n<<<<<<<<<< RAW OUTPUT START >>>>>>>>>>\n{output_string}\n<<<<<<<<<<< RAW OUTPUT END >>>>>>>>>>>")
        
        if not output_string:
            logger.error("--- [ERROR] LLM의 최종 응답이 비어있습니다.")
            # 비어있는 경우의 에러 처리를 명확하게 합니다.
            return json.loads('{"chatType": "error", "answer": "죄송합니다, 답변을 생성하는 데 실패했습니다. 응답이 비어있습니다."}')



        # 최종 결과에서 ```json ... ``` 부분을 추출
        clean_json_string = ""

        # 1. 먼저 마크다운 블록(```json ... ```)이 있는지 확인하고, 있다면 내부의 JSON만 추출합니다.
        logger.info("--- [STEP 5] 정규식을 사용해 JSON 블록 찾는 중... ---")
        match = re.search(r"```(json)?\s*(\{.*?\})\s*```", output_string, re.DOTALL)
        
        if match:
            clean_json_string = match.group(2).strip()
            logger.info("--- [STEP 6a] 마크다운 블록에서 JSON을 성공적으로 추출했습니다. ---")
        else:
            # 만약 ```json ``` 마크다운을 생성하지 않을 시 전체 문자열 사용 (LLM이 지시를 완전히 따르지 않은 경우일 수 있음)
            logger.warning("--- [STEP 6b] JSON 블록을 찾지 못했습니다. 전체 문자열을 사용합니다. ---")
            clean_json_string = output_string.strip()


        # --- 디버깅 코드 추가 ---
        logger.info("--- [STEP 7] 파싱할 최종 JSON 문자열(Cleaned JSON)을 준비했습니다. ---")
        logger.info(f"\n<<<<<<<<<< CLEAN JSON START >>>>>>>>>>\n{clean_json_string}\n<<<<<<<<<<< CLEAN JSON END >>>>>>>>>>>")
        

        logger.info(f"--- [STEP 8] json.loads()로 문자열을 파싱 시도 중... ---")
        parsed_data = json.loads(clean_json_string)
        
        logger.info(f"--- [STEP 9] json.loads()가 정상적으로 완료되었습니다. 데이터 타입: {type(parsed_data)} ---")
        
        # 마지막 단계: 이 로그가 찍히면, 함수 자체는 성공적으로 끝난 것입니다.
        logger.info("--- ✅ [마지막 단계] 모든 처리가 완료되었습니다. 이제 파싱된 딕셔너리를 반환합니다. ---")
        return parsed_data
    
    

    except Exception as e:
        logger.error(f"--- 🚨 [예외 발생] 예외가 발생했습니다: {e}", exc_info=True)
        raise e