from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages.tool import ToolCall
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Optional
import operator
from langchain_core.runnables import RunnableConfig

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
    model="gemini-2.5-pro", 
    temperature=0, 
    google_api_key=GEMINI_API_KEY,
    timeout=60,  # <-- ⭐️ 이 부분을 추가해 주세요! (단위: 초)
)


# 프롬프트 1: 도구 선택 전용 - 채팅 히스토리 컨텍스트 지원
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 요청 의도를 정확히 분석하여, 어떤 도구를 어떻게 호출할지 결정하는 전문가입니다.

    ---
    ### **1단계: 사용자 의도 분석 및 도구 결정**
     
    - 사용자의 메시지에 **"[사용자가 이미지를 첨부했습니다]"** 라는 텍스트가 포함되어 있다면, 다른 텍스트 내용과 상관없이 **반드시** `search_ingredient_by_image` 도구를 호출해야 합니다. 이때 `image_b64` 인자는 빈 문자열("")로 호출하세요. 시스템이 나중에 실제 데이터를 채워 넣을 것입니다.
    - 사용자의 메시지에 **YouTube URL**이 포함되어 있다면, `extract_recipe_from_youtube` 도구를 호출하세요.
    - '상품 정보'나 '구매' 관련 텍스트 요청은 `search_ingredient_by_text` 도구를 호출하세요.
    - 그 외의 모든 '요리 관련 대화'는 `text_based_cooking_assistant` 도구를 호출하세요.

    ---
    ### **2단계: 도구 호출 규칙**

    - 사용자가 **여러 요리 레시피**를 한 번에 요청했다면(예: "김치찌개랑 된장찌개 레시피"), 반드시 `text_based_cooking_assistant` 도구를 **요리별로 각각** 호출해야 합니다.
    """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
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




#-----------------------------------------------------------------------------------------------
# create_tool_calling_agent는 LLM이 도구 사용을 '결정'하게 만드는 부분입니다.
# 이 부분은 LangChain Core 기능이므로 변경이 없습니다.
# agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)
    
# 4. LangGraph 구성
# LangGraph 패턴에 더 적합하도록, LLM에 도구를 바인딩하고 프롬프트와 연결합니다.
llm_with_tools = llm.bind_tools(tools)
agent_runnable = tool_calling_prompt | llm_with_tools
# 에이전트가 작업하는 동안 유지하고 업데이트할 데이터 구조입니다.
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    # 👇 이 필드를 추가해주세요!
    image_b64: Optional[str]
    tool_output: Optional[str]


# 2. LangGraph의 노드(Node)와 엣지(Edge) 정의
# 1). 도구 선택 노드 - 채팅 히스토리 컨텍스트 지원
async def select_tool(state):
    logger.info("--- [LangGraph] 🧠 Node (select_tool) 실행 ---")
    messages = state["messages"]
    logger.info(f"--- [LangGraph] LLM에 전달할 메시지: {messages} ---")

    try:
        logger.info("--- [LangGraph] ⏳ agent_runnable.ainvoke 호출 시작... (이 단계에서 외부 API 호출로 인해 시간이 소요될 수 있습니다) ---")
        response_message = await agent_runnable.ainvoke({
            "messages": messages,
            "agent_scratchpad": []
        })
        logger.info("--- [LangGraph] ✅ agent_runnable.ainvoke 호출 완료. ---")
        logger.info(f"--- [LangGraph] 도구 선택 결과: {response_message} ---")
        # 기존 메시지 기록에 새로운 응답 메시지를 추가하여 반환합니다.
        # 이렇게 해야 대화의 전체 맥락이 유지됩니다.
        return {"messages": state["messages"] + [response_message]}
    
    except Exception as e:
        logger.error(f"--- [LangGraph] 🚨 agent_runnable.ainvoke 호출 중 심각한 오류 발생! 🚨 ---")
        # 💡 여기가 핵심! 예외 객체 'e'와 함께 스택 트레이스를 함께 로깅합니다.
        # 'MALFORMED_FUNCTION_CALL' 같은 오류의 경우, LLM이 생성하려 했던 텍스트가 여기에 포함될 수 있습니다.
        logger.error(f"오류 내용: {e}", exc_info=True)
        
        # 이 예시에서는 예외를 다시 발생시켜 실행을 중단합니다.
        raise e


# 2). 이미지 데이터 주입 노드 - 실제 도구(tool_node)를 실행하기 전에 AgentState에 저장해 둔 이미지 데이터를 도구 호출 정보에 주입하는 중간 다리 역할을 하는 노드.
def inject_image_data(state: AgentState) -> dict:
    """
    select_tool 단계에서 생성된 도구 호출(tool_calls)을 확인하고,
    search_ingredient_by_image 호출이 있다면 state에 저장된 image_b64 데이터를 주입합니다.
    변경이 필요 없거나 불가능한 경우, 빈 딕셔너리를 반환하여 상태를 그대로 유지합니다.
    """
    logger.info("--- [LangGraph] 💉 Node (inject_image_data) 실행 ---")

    image_to_inject = state.get("image_b64")
    logger.info(f"--- [LangGraph] 💉 주입할 이미지 데이터: {image_to_inject} ---")

    if not image_to_inject:
        logger.warning("--- [LangGraph] 💉 주입할 이미지 데이터가 없습니다 ---")
        return {**state, "messages": state["messages"]}
    
    # 가장 마지막 메시지 (AIMessage)를 가져옵니다.
        logger.warning("--- [LangGraph] 💉 주입할 이미지 데이터가 없으므로, 아무 작업도 수행하지 않습니다. ---")
        return {} # 변경 사항 없음을 명시적으로 알림
    last_message = state["messages"][-1]
    # logger.info(f"--- [LangGraph] 💉 마지막 메시지: {last_message} ---")
    
    # # state에 이미지 데이터가 있고, 마지막 메시지에 tool_calls가 있을 때만 작동
    # if image_to_inject and last_message.tool_calls:
    #     # 새로운 tool_calls 리스트를 만듭니다.
    #     new_tool_calls = []
    #     for tool_call in last_message.tool_calls:
    #         # 이미지 검색 도구일 경우
    #         if tool_call["name"] == "search_ingredient_by_image":
    #             logger.info(f"--- [LangGraph] 👉 search_ingredient_by_image 호출에 이미지 데이터 주입 ---")
    #             # 기존 args를 복사하고 image_b64 값을 덮어씁니다.
    #             new_args = tool_call["args"].copy()
    #             new_args["image_b64"] = image_to_inject

    #             # 💡 [수정] 일반 딕셔너리 대신 LangChain의 공식 ToolCall 객체를 생성하여 안정성을 높입니다.
    #             new_tool_calls.append(
    #                 ToolCall(name=tool_call["name"], args=new_args, id=tool_call["id"])
    #             )
    #         else:
    #             # 다른 도구는 그대로 추가
    #             new_tool_calls.append(tool_call)
        
    #     # 1. 기존 메시지 리스트에서 마지막 AI Message를 제거합니다.
    #     messages_without_last = state["messages"][:-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("--- [LangGraph] 💉 마지막 메시지에 tool_calls가 없으므로, 아무 작업도 수행하지 않습니다. ---")
        return {} # 변경 사항 없음

    needs_update = False
    new_tool_calls = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search_ingredient_by_image":
            logger.info(f"--- [LangGraph] 👉 search_ingredient_by_image 호출에 이미지 데이터 주입 ---")
            needs_update = True
            new_args = tool_call["args"].copy()
            new_args["image_b64"] = image_to_inject
            new_tool_calls.append(
                ToolCall(name=tool_call["name"], args=new_args, id=tool_call["id"])
            )
        else:
            # 다른 도구는 그대로 추가
            new_tool_calls.append(tool_call)

    # 이미지 주입이 실제로 일어났을 때만 메시지 리스트를 교체합니다.
    if needs_update:
        logger.info("--- [LangGraph] 💉 이미지 데이터 주입 완료. 메시지 상태를 업데이트합니다. ---")

        # 2. 새로운 tool_calls로 완전히 새로운 AIMessage 객체를 만듭니다.
        new_ai_message = AIMessage(
            content=last_message.content,
            tool_calls=new_tool_calls,
            id=last_message.id,
        )

         # 3. 메시지 리스트에 새로운 AI Message를 추가합니다.
        final_messages = state["messages"][:-1] + [new_ai_message]
        
        logger.info(f"--- [LangGraph] 👉 데이터 주입 후 최종 메시지 상태: {final_messages}")
        
        # 4. LangGraph에 수정된 'messages'만 반환합니다.
        return {"messages": final_messages}

    # 주입할 도구를 찾지 못한 경우
    logger.warning("--- [LangGraph] 💉 이미지 주입이 필요한 도구 호출을 찾지 못했습니다. ---")
    return {}


# # 3). Tool 노드: 미리 만들어진 ToolNode를 사용합니다.
# tool_node = ToolNode(tools)
# 3). [변경] 직접 만드는 Tool 실행 노드
async def custom_tool_node(state: AgentState, config: RunnableConfig):
    """
    ToolNode를 대체하는 커스텀 노드.
    Tool 실행 후 ToolMessage를 생성하는 대신, 결과 문자열을 state['tool_output']에 저장합니다.
    """
    logger.info("--- [LangGraph] 🛠️ Node (custom_tool_node) 실행 ---")
    
    # 마지막 메시지에서 tool_calls를 가져옵니다.
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    # Tool 이름과 인자를 매핑한 딕셔너리를 만듭니다.
    tool_map = {tool.name: tool for tool in tools}

    # 각 tool_call을 비동기적으로 실행합니다.
    call_responses = []
    for call in tool_calls:
        tool_to_call = tool_map.get(call["name"])
        if tool_to_call:
            # config를 함께 전달해주는 것이 좋습니다 (LangSmith 추적 등에 사용될 수 있음)
            response = await tool_to_call.ainvoke(call["args"], config=config)
            call_responses.append(response)
        else:
            call_responses.append(f"Error: Tool '{call['name']}' not found.")
    
    # 여기서는 Tool이 하나만 호출된다고 가정하고 첫 번째 결과만 사용합니다.
    # 여러 Tool 호출을 처리하려면 이 부분을 수정해야 합니다.
    tool_output_str = call_responses[0] if call_responses else ""

    logger.info(f"--- [LangGraph] 🛠️ Tool 실행 완료. 결과: {tool_output_str[:300]}... ---")

    # ⭐️ 핵심: messages 리스트에 ToolMessage를 추가하는 대신,
    # state의 'tool_output' 필드에 직접 결과 문자열을 저장합니다.
    return {"tool_output": tool_output_str}


# 4). 최종 답변 생성 노드
async def generate_final_answer(state):
    logger.info("--- [LangGraph] ✍️ Node (generate_final_answer) 실행 ---")

     # ⭐️ 핵심: 더 이상 MessagesPlaceholder를 사용하지 않습니다.
    # state['tool_output']에 저장된 깨끗한 결과 문자열만 프롬프트에 직접 주입합니다.
    
    # 간단한 문자열 기반 프롬프트
    final_prompt = ChatPromptTemplate.from_template(
        """
        당신은 주어진 도구의 결과(Tool Output)를 분석하여, 정해진 규칙에 따라 최종 JSON으로 완벽하게 변환하는 JSON 포맷팅 전문가입니다.
        당신의 유일한 임무는 JSON을 생성하는 것입니다. **절대로 다른 도구를 호출하지 마세요.**
        아래에 제공된 "Tool Output" 내용을 바탕으로 JSON을 생성하세요.

        ---
        ### Tool Output:
        {tool_output}
        ---

        ### JSON 생성 규칙:
        
        #### 이미지 검색 결과 처리:
        - `search_ingredient_by_image` 호출 결과는 `cart` 형식 JSON으로 변환
        - 각 결과는 반드시 `product_name`, `price`, `image_url`, `product_address` 포함

        #### 1. `chatType` 결정:
        - Tool Output에 'product_name'과 'price'가 포함되어 있으면 `chatType`은 "cart"입니다.
        - 그 외의 경우(레시피 등) `chatType`은 "chat"입니다.

        #### 2. 최종 JSON 구조 (규칙은 이전과 동일):
        - `chatType`이 "cart"일 경우: `product_name`, `price`, `image_url`, `product_address` 4개의 키만 정확히 추출하여 `ingredients` 리스트 생성.
        - `chatType`이 "chat"일 경우: `answer`, `recipes`, `ingredients`, `recipe` 구조에 맞게 생성.
        """
    )



    # 'JSON 생성' 역할을 수행하는 체인을 구성합니다.
    chain = final_prompt | llm

    # 수정된 프롬프트에 맞춰, 'messages'라는 키로 전체 대화 기록을 전달합니다.
    # final_response = await chain.ainvoke({"messages": state["messages"]})
    final_response_msg = await chain.ainvoke({"tool_output": state["tool_output"]})
    logger.info(f"--- [LangGraph] ✍️ 최종 응답: {final_response_msg} ---")
    
    # 기존 메시지 기록에 최종 응답을 추가하여 전체 대화 기록을 업데이트합니다.
    # ⭐️ 최종 결과 AIMessage를 messages 리스트에 추가하여 전체 대화 기록을 완성합니다.
    return {"messages": state["messages"] + [final_response_msg]}


def should_call_tool(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "action"
    return END


# 채팅 히스토리를 LangGraph 메시지 형식으로 변환하는 함수
# def convert_chat_history_to_messages(chat_history: list) -> list:
#     """
#     프론트엔드에서 받은 채팅 히스토리를 LangGraph 메시지 형식으로 변환합니다.
    
#     [처리 가능한 형식 1: 텍스트 메시지]
#     [
#         {"role": "user", "content": "라멘 레시피 알려줘"},
#         {"role": "assistant", "content": "요청하신 레시피입니다."},
#         {"role": "user", "content": "볶음밥 레시피 알려줘"},
#         {"role": "assistant", "content": "어떤 볶음밥 레시피를 원하시나요?\n\n1. 김치 볶음밥\n2. 새우 볶음밥\n3. 게살 볶음밥\n4. 파인애플 볶음밥\n\n다른 원하시는 볶음밥 종류가 있으시면 말씀해주세요!"},
#         {"role": "user", "content": "4번"}
#     ]

#     [처리 가능한 형식 2: 이미지 + 텍스트 멀티모달 메시지]
#     {
#         "role": "user", 
#         "content": [
#             {"type": "text", "text": "이 이미지 분석해서 재료 찾아줘"},
#             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
#         ]
#     }
#     """
#     messages = []
    
#     for msg in chat_history:
#         role = msg.get("role", "").lower()
#         content = msg.get("content", "")
        
#         if not content:
#             continue
            
#         if role == "user":
#             messages.append(HumanMessage(content=content))    # content가 문자열이든 리스트든 HumanMessage가 알아서 처리합니다.
#         elif role == "assistant":
#             messages.append(AIMessage(content=content))   # AI 메시지는 항상 텍스트이므로 그대로 사용합니다.
#         else:
#             # 알 수 없는 role은 user로 처리
#             logger.warning(f"알 수 없는 role '{role}', user로 처리합니다.")
#             messages.append(HumanMessage(content=content))
    
#     logger.info(f"채팅 히스토리를 {len(messages)}개의 LangGraph 메시지로 변환했습니다.")
#     return messages


# 4. 그래프(Graph) 생성 및 연결
# 상태 그래프를 만들고 위에서 정의한 노드와 엣지를 연결합니다.
workflow = StateGraph(AgentState)

# 1️⃣ 노드들을 먼저 그래프에 '등록'합니다.
workflow.add_node("agent", select_tool)
workflow.add_node("image_injector", inject_image_data) # 💉 새로운 노드 등록
# workflow.add_node("action", tool_node)
workflow.add_node("action", custom_tool_node)
workflow.add_node("formatter", generate_final_answer)

# 2️⃣ 그래프의 시작점을 'agent' 노드로 설정합니다.
workflow.set_entry_point("agent")

# 3️⃣ '등록된' 노드들 사이의 연결선을 정의하는 조건부 엣지를 추가합니다. 'agent' 노드 다음에 should_continue 함수를 실행하여
# 'action'으로 갈지, 'END'로 갈지 결정합니다. action으로 가기 전에 image_injector를 거치도록 합니다.
workflow.add_conditional_edges(
    "agent",
    should_call_tool,
    {
        "action": "image_injector", # 💉 'action' 대신 'image_injector'로 변경
        END: END,
    },
)

# 4️⃣ 엣지를 다시 연결합니다.
workflow.add_edge("image_injector", "action") # 💉 image_injector -> action
workflow.add_edge("action", "formatter")
workflow.add_edge("formatter", END)

# 5️⃣ 그래프를 컴파일합니다.
app = workflow.compile()



async def run_agent(input_data: dict):
    """
    사용자 입력을 받아 에이전트를 실행하고 결과를 반환합니다.
    input_data: {"message": str, "image_b64": Optional[str]}
    """
    logger.info("--- [STEP 0] Agent Start ---")
    
    try:
        # 단일 메시지 처리 (기존 방식)
        user_message = input_data.get("message", "")
        image_bytes = input_data.get("image")
        # image_content_type = input_data.get("image_content_type", "image/jpeg")
        logger.info(f"--- [STEP 1b] 단일 메시지 처리: {user_message} ---")

        inputs = {} # inputs 딕셔너리를 먼저 초기화
        
        if image_bytes:
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            # logger.info(f"--- image_b64: {image_b64} ---")
            logger.info("--- [STEP 1c] 이미지 데이터가 포함되었습니다. ---")


            # ⭐️⭐️⭐️ [모든 문제의 원흉을 해결하는 코드] ⭐️⭐️⭐️
            # 사용자가 텍스트 없이 이미지만 보냈을 경우(user_message가 None이거나 비어있을 때),
            # 대화의 문맥을 만들어주기 위한 기본 메시지를 설정합니다.
            if not user_message:
                user_message = "이 이미지에 있는 상품 정보를 찾아줘."
                logger.info(f"--- [STEP 1d] 텍스트가 없어 기본 메시지 설정: '{user_message}' ---")
            
            # 💡 [수정] HumanMessage의 content를 복잡한 리스트가 아닌 단순 문자열로 만듭니다.
            # 도구 선택 프롬프트가 인식할 수 있도록 이미지 첨부 사실을 텍스트에 명시적으로 추가합니다.
            full_message = f"{user_message} [사용자가 이미지를 첨부했습니다]"
            messages = [HumanMessage(content=full_message)]
            inputs = {"messages": messages, "image_b64": image_b64} # 실제 데이터는 state에 저장하여 LangGraph로 전달합니다.

        else:
            # 이미지가 없거나 텍스트만 있는 경우
            messages = [HumanMessage(content=user_message or "")] # user_message가 None일 경우 빈 문자열로 처리
            inputs = {"messages": messages, "image_b64": None} # image_b64는 None으로 전달
        
        if not messages:
             raise ValueError("처리할 메시지가 없습니다.")
        
        # LangGraph 실행
        logger.info("--- [STEP 2] app.ainvoke 호출 중... ---")
        logger.info(f"====== inputs 확인 : {inputs} ======")
        result_state = await app.ainvoke(inputs)
        logger.info("--- [STEP 3] app.ainvoke가 정상적으로 완료되었습니다. ---")

        # 결과에서 최종 AI 응답 메시지를 추출합니다.
        # output_string = result.get("output", "")
        final_message = result_state["messages"][-1]
        output_string = final_message.content if isinstance(final_message, AIMessage) else ""

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