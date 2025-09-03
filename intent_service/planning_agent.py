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
import base64
import asyncio
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# .env 파일 로드 및 API 키 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# API 키 검증
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# LangChain이 찾는 환경 변수로도 설정
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# 다른 폴더에 있는 모듈을 가져오기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# 부모 디렉토리(프로젝트의 루트 폴더) 경로를 가져옵니다.
project_root = os.path.dirname(current_dir)

# 파이썬이 모듈을 검색하는 경로 리스트에 프로젝트 루트 폴더를 추가합니다.
sys.path.append(project_root)

# 내부 처리 함수들을 import
from text_service.agent.core import TextAgent
from video_service.core.extractor import extract_recipe_from_youtube_internal
from ingredient_service.core import IngredientProcessor

# 내부 처리 인스턴스 생성
text_agent = TextAgent()
ingredient_processor = IngredientProcessor()

# 내부 처리 함수들 정의
async def text_based_cooking_assistant_internal(query: str) -> str:
    """텍스트 기반 요리 어시스턴트 - 내부 처리"""
    try:
        logger.info(f"내부 텍스트 처리 시작: {query}")
        result = await text_agent.process_message(query)
        logger.info(f"내부 텍스트 처리 완료: {result}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"내부 텍스트 처리 오류: {e}")
        return json.dumps({"error": f"텍스트 처리 오류: {str(e)}"}, ensure_ascii=False)

async def extract_recipe_from_youtube_internal_wrapper(youtube_url: str) -> str:
    """유튜브 레시피 추출 - 내부 처리"""
    try:
        logger.info(f"내부 비디오 처리 시작: {youtube_url}")
        result = await extract_recipe_from_youtube_internal(youtube_url)
        logger.info(f"내부 비디오 처리 완료: {result}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"내부 비디오 처리 오류: {e}")
        return json.dumps({"error": f"비디오 처리 오류: {str(e)}"}, ensure_ascii=False)

async def search_ingredient_by_text_internal(query: str) -> str:
    """재료 텍스트 검색 - 내부 처리"""
    try:
        logger.info(f"내부 재료 텍스트 검색 시작: {query}")
        result = await ingredient_processor.search_by_text(query)
        logger.info(f"내부 재료 텍스트 검색 완료: {result}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"내부 재료 텍스트 검색 오류: {e}")
        return json.dumps({"error": f"재료 검색 오류: {str(e)}"}, ensure_ascii=False)

async def search_ingredient_by_image_internal(image_b64: str) -> str:
    """재료 이미지 검색 - 내부 처리"""
    try:
        logger.info("내부 재료 이미지 검색 시작")
        result = await ingredient_processor.search_by_image(image_b64)
        logger.info(f"내부 재료 이미지 검색 완료: {result}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"내부 재료 이미지 검색 오류: {e}")
        return json.dumps({"error": f"이미지 검색 오류: {str(e)}"}, ensure_ascii=False)

# 기존 인터페이스 유지를 위한 래퍼 함수들 (하위 호환성)
from langchain_core.tools import tool

@tool
async def text_based_cooking_assistant(query: str) -> str:
    """텍스트 기반의 요리 관련 질문에 답변할 때 사용합니다."""
    return await text_based_cooking_assistant_internal(query)

@tool
async def extract_recipe_from_youtube(youtube_url: str) -> str:
    """유튜브 영상에서 레시피를 추출할 때 사용합니다."""
    return await extract_recipe_from_youtube_internal_wrapper(youtube_url)

@tool
async def search_ingredient_by_text(query: str) -> str:
    """사용자가 상품 정보를 찾거나, 장바구니에 상품을 담으려 할 때 사용합니다."""
    return await search_ingredient_by_text_internal(query)

@tool
async def search_ingredient_by_image(image_b64: str) -> str:
    """사용자가 '이미지'만으로 재료나 상품 구매 정보를 물어볼 때 사용합니다."""
    return await search_ingredient_by_image_internal(image_b64)

# 1. 사용할 도구(Tools) 정의 - 내부 처리 함수들 사용
tools = [
    text_based_cooking_assistant,
    extract_recipe_from_youtube,
    search_ingredient_by_text,  # 내부 처리로 변경됨
    search_ingredient_by_image, # 내부 처리로 변경됨
]

# 2. LLM 모델 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    timeout=60,
    max_tokens=8192,  # 토큰 제한 명시
)

# 프롬프트 1: 도구 선택 전용 - 채팅 히스토리 컨텍스트 지원
tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    당신은 사용자의 요청 의도를 정확히 분석하여, 어떤 도구를 어떻게 호출할지 결정하는 전문가입니다.

    ---
    ### **1단계: 사용자 의도 분석 및 도구 결정**
     
    - 사용자의 메시지에 **"[사용자가 이미지를 첨부했습니다]"** 라는 텍스트가 포함되어 있다면, 다른 텍스트 내용과 상관없이 **반드시** `search_ingredient_by_image` 도구를 호출해야 합니다. 이때 `image_b64` 인자는 빈 문자열("")로 호출하세요. 시스템이 나중에 실제 데이터를 채워 넣을 것입니다.
    - 사용자의 메시지에 **YouTube URL**이 포함되어 있다면, `extract_recipe_from_youtube` 도구를 호출하세요.
    - '상품 정보'나 '구매' 관련 텍스트 요청은 `search_ingredient_by_text` 도구를 호출하세요. query에는 상품 이름을 넣으세요

    - 그 외의 모든 '요리 관련 대화'는 `text_based_cooking_assistant` 도구를 호출하세요.

    ---
    ### **2단계: 도구 호출 규칙**

    - 사용자가 **여러 요리 레시피**를 한 번에 요청했다면(예: "김치찌개랑 된장찌개 레시피"), 반드시 `text_based_cooking_assistant` 도구를 **요리별로 각각** 호출해야 합니다.
    - 사용자의 메시지에 **YouTube URL과 요리 관련 텍스트가 함께 포함**되어 있다면, `extract_recipe_from_youtube`와 `text_based_cooking_assistant`를 **모두** 호출하세요. 각 호출은 별도의 tool_call로 생성합니다.
    - 텍스트에 **여러 요리명이 함께 포함**된 경우에는, **가장 명확하거나 최근에 언급된 순으로 최대 2개까지만** 선택하여 `text_based_cooking_assistant`를 각각 호출하세요. 3개 이상이면 **상위 2개만** 선택하세요.
    - 위 규칙으로 생성하는 각 `text_based_cooking_assistant` 호출의 `query`는 **요리명만 간결하게 포함**한 문장으로 하세요. 예: "김치찌개 레시피 알려줘"
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 3. 프롬프트(Prompt) 설정 - 에이전트에게 내리는 지시사항
json_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
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

    - `tool_output`에 상품 정보가 포함된 경우면 `chatType`은 "cart"입니다.
    - `tool_output`에 **완전한 레시피 정보**(ingredients와 recipe가 모두 존재하고 비어있지 않음)가 포함된 경우면 `chatType`은 "recipe"입니다.
    - **선택지 제공이나 사용자에게 답변을 요청하는 경우**(예: "어떤 볶음밥을 원하시나요?", "번호나 요리명을 말씀해주세요")는 `chatType`을 "chat"으로 설정하세요.
    - 그 외의 경우(예: 사용자의 답변 요청, 불완전한 정보) `chatType`은 "chat"입니다.


    #### **2. 최종 JSON 구조:**

    - **`chatType`이 "recipe"일 경우:**
      - `tool_output`의 `answer`를 참고하여 친절한 답변을 생성하세요.
      - `ingredients`는 반드시 **item, amount, unit**을 키로 가지는 객체의 리스트여야 합니다. amount나 unit이 없는 경우(예: '얼음 약간')에는 빈 문자열("")을 값으로 채워주세요.
      - `recipes` 리스트를 `tool_output`의 내용으로 채우세요.
      - 최종 구조는 반드시 아래와 같아야 합니다.
      ```json
      {{
        "chatType": "recipe",
        "answer": "사용자 채팅에 답변하는 내용.(이모지 사용)",
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
        "answer": "사용자 채팅에 답변하는 내용.(이모지 사용)",
        "recipes": [
          {{
            "source": "ingredient_search",
            "food_name": "사용자가 검색한 상품명",
            "product": [
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
     
    - **`chatType`이 "chat"일 경우:**
    ```json
      {{
        "chatType": "chat",
        "answer": "사용자에게 다시 답변하는 내용.",
        "recipes": [],
      }}
    ```
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 4. LangGraph 구성
llm_with_tools = llm.bind_tools(tools)
agent_runnable = tool_calling_prompt | llm_with_tools

# 에이전트가 작업하는 동안 유지하고 업데이트할 데이터 구조입니다.
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    image_b64: Optional[str]
    tool_output: Optional[str]

# 2. LangGraph의 노드(Node)와 엣지(Edge) 정의
# 1). 도구 선택 노드 - 채팅 히스토리 컨텍스트 지원
async def select_tool(state):
    messages = state["messages"]

    try:
        response_message = await agent_runnable.ainvoke(
            {"messages": messages, "agent_scratchpad": []}
        )
        return {"messages": state["messages"] + [response_message]}

    except Exception as e:
        raise e

# 2). 이미지 데이터 주입 노드
def inject_image_data(state: AgentState) -> dict:
    """
    select_tool 단계에서 생성된 도구 호출(tool_calls)을 확인하고,
    search_ingredient_by_image 호출이 있다면 state에 저장된 image_b64 데이터를 주입합니다.
    """
    logger.info("--- [LangGraph] 💉 Node (inject_image_data) 실행 ---")

    image_to_inject = state.get("image_b64")
    if not image_to_inject:
        logger.warning("--- [LangGraph] 💉 주입할 이미지 데이터가 없습니다 ---")
        return {**state, "messages": state["messages"]}
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning(
            "--- [LangGraph] 💉 마지막 메시지에 tool_calls가 없으므로, 아무 작업도 수행하지 않습니다. ---"
        )
        return {}

    needs_update = False
    new_tool_calls = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search_ingredient_by_image":
            logger.info(
                f"--- [LangGraph] 👉 search_ingredient_by_image 호출에 이미지 데이터 주입 ---"
            )
            needs_update = True
            new_args = tool_call["args"].copy()
            new_args["image_b64"] = image_to_inject
            new_tool_calls.append(
                ToolCall(name=tool_call["name"], args=new_args, id=tool_call["id"])
            )
        else:
            new_tool_calls.append(tool_call)

    if needs_update:
        logger.info(
            "--- [LangGraph] 💉 이미지 데이터 주입 완료. 메시지 상태를 업데이트합니다. ---"
        )
        new_ai_message = AIMessage(
            content=last_message.content,
            tool_calls=new_tool_calls,
            id=last_message.id,
        )
        final_messages = state["messages"][:-1] + [new_ai_message]

        logger.info(
            f"--- [LangGraph] 👉 데이터 주입 후 최종 메시지 상태: {final_messages}"
        )

        return {"messages": final_messages}

    logger.warning(
        "--- [LangGraph] 💉 이미지 주입이 필요한 도구 호출을 찾지 못했습니다. ---"
    )
    return {}

# 3). Tool 노드: 내부 처리 방식으로 변경
async def custom_tool_node(state: AgentState, config: RunnableConfig):
    """
    ToolNode를 대체하는 커스텀 노드 - 내부 처리 방식
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    # 내부 처리 함수 매핑
    tool_map = {
        "text_based_cooking_assistant": text_based_cooking_assistant_internal,
        "extract_recipe_from_youtube": extract_recipe_from_youtube_internal_wrapper,
        "search_ingredient_by_text": search_ingredient_by_text_internal,
        "search_ingredient_by_image": search_ingredient_by_image_internal,
    }

    # 각 tool_call을 비동기적으로 실행하고 모두 취합
    call_tasks = []
    for call in tool_calls:
        tool_func = tool_map.get(call["name"])
        if tool_func:
            call_tasks.append(tool_func(**call["args"]))
        else:
            call_tasks.append(asyncio.sleep(0, result=f"Error: Tool '{call['name']}' not found."))

    call_responses = await asyncio.gather(*call_tasks)

    # 단일 응답이면 문자열 그대로, 다건이면 리스트로 반환
    if len(call_responses) == 1:
        return {"tool_output": call_responses[0]}
    return {"tool_output": call_responses}

# 'JSON 생성' 역할을 수행하는 체인을 미리 구성합니다.
final_prompt_for_formatter = ChatPromptTemplate.from_template(
    """
    당신은 주어진 도구의 결과(Tool Output)를 분석하여, 정해진 API 규격에 따라 최종 JSON으로 완벽하게 변환하는 JSON 포맷팅 전문가입니다.
    당신의 유일한 임무는 JSON을 생성하는 것입니다. **절대로 다른 도구를 호출하거나 불필요한 설명을 추가하지 마세요.**
    오직 JSON 객체만 생성해야 합니다.

    ---
    ### Tool Output (내부 처리에서 온 원본 데이터):
    {tool_output}
    ---

    ### JSON 생성 규칙:

    #### 1. `chatType` 결정:
    - Tool Output에 상품 정보가 있다면 'cart' 로 설정하세요
    - Tool Output에 **완전한 레시피 정보**(ingredients와 recipe가 모두 존재하고 비어있지 않음)가 있다면 'recipe'로 설정하세요.
    - **선택지 제공이나 사용자에게 답변을 요청하는 경우**(예: "어떤 볶음밥을 원하시나요?", "번호나 요리명을 말씀해주세요")는 `chatType`을 "chat"으로 설정하세요.
    - 그 외의 경우(예: 사용자의 답변 요청, 불완전한 정보) `chatType`은 "chat"입니다.

    #### 2. 최종 JSON 구조 (규칙은 이전과 동일):
    - `chatType`이 **"cart"**일 경우, 아래 규칙을 철저히 따르세요.

    - **[핵심 추출 규칙]**: Tool Output의 `results` 리스트에 있는 각 상품 객체에서 **`product_name`, `price`, `image_url`, `product_address`** 4개의 키와 값만 정확히 추출하세요. 다른 모든 필드(id, category, quantity, similarity_score 등)는 **반드시 제외**해야 합니다.
    - 추출한 4개의 키로 구성된 객체들의 리스트를 만드세요.
    - 이 리스트를 최종 JSON의 `recipes[0].product` 키의 값으로 사용하세요.
    - `answer` 필드에는 "요청하신 상품을 찾았습니다."와 같은 간단한 안내 문구를 넣으세요.
    - `food_name` 필드에는 상품들의 공통 카테고리나 주요 특징을 반영한 의미있는 대제목을 생성하세요. 예를 들어, "감자", "토마토", "돼지고기" 등과 같이 상품의 성격을 잘 나타내는 제목을 만드세요.

    - 최종 결과는 반드시 아래의 JSON 구조와 일치해야 합니다.

    ```json
    {{
      "chatType": "cart",
      "answer": "사용자에게 답변할 내용",
      "recipes": [
        {{
          "source": "ingredient_search",
          "food_name": "상품 대제목",
          "product": [
            {{
              "product_name": "상품 이름 1",
              "price": 1234,
              "image_url": "https://...",
              "product_address": "https://..."
            }},
            {{
              "product_name": "상품 이름 2",
              "price": 5678,
              "image_url": "https://...",
              "product_address": "https://..."
            }}
          ],
          "recipe": []
        }}
      ]
    }}
    ```
    """
)
formatter_chain = final_prompt_for_formatter | llm

# 4). 최종 답변 생성 노드 (최적화됨)
async def generate_final_answer(state):
    logger.info("--- [LangGraph] ✍️ Node (generate_final_answer) 실행 ---")

    try:
        # tool_output이 이미 완성된 레시피 데이터인 경우 바로 사용
        tool_output = state.get("tool_output") or ""

        # 툴 호출이 전혀 없었던 경우(비요리 대화 등) 기본 OTHER 응답 반환
        if not tool_output:
            from langchain_core.messages import AIMessage
            default_json = {
                "chatType": "chat",
                "answer": "무엇을 도와드릴까요? 요리/레시피 관련 요청을 말씀해 주세요.",
                "recipes": []
            }
            import json as _json
            return {"messages": state["messages"] + [AIMessage(content=f"```json\n{_json.dumps(default_json, ensure_ascii=False)}\n```")]} 
        
        if isinstance(tool_output, str):
            # JSON 문자열인 경우 파싱
            import json
            try:
                parsed_output = json.loads(tool_output)
            except:
                # 파싱 실패시 기존 방식 사용
                final_response_msg = await formatter_chain.ainvoke(
                    {"tool_output": tool_output}
                )
                logger.info(f"--- [LangGraph] ✍️ 최종 응답 (기존 방식): {final_response_msg} ---")
                return {"messages": state["messages"] + [final_response_msg]}
        elif isinstance(tool_output, list):
            # 다건 응답: 각 항목을 개별 파싱 후 병합 준비
            import json
            parsed_list = []
            for item in tool_output:
                if isinstance(item, str):
                    try:
                        parsed_list.append(json.loads(item))
                    except Exception:
                        parsed_list.append(item)
                else:
                    parsed_list.append(item)
            parsed_output = parsed_list
        else:
            parsed_output = tool_output
        
        # 이미 완성된 단건 레시피 데이터를 바로 JSON으로 변환
        if isinstance(parsed_output, dict) and ("source" in parsed_output or "food_name" in parsed_output):
            # 레시피 데이터가 이미 완성된 경우
            # source가 없으면 추가
            recipe_data = parsed_output.copy()
            if "source" not in recipe_data:
                recipe_data["source"] = "text"
            
            # 선택지 제공 상황인지 확인 (ingredients와 recipe가 비어있고 answer에 선택지가 있는 경우)
            answer_text = recipe_data.get("answer", "")
            has_ingredients = recipe_data.get("ingredients") and len(recipe_data.get("ingredients", [])) > 0
            has_recipe = recipe_data.get("recipe") and len(recipe_data.get("recipe", [])) > 0
            is_choice_providing = ("번호" in answer_text or "말씀해" in answer_text or "원하시나요" in answer_text) and not (has_ingredients and has_recipe)
            
            if is_choice_providing:
                # 선택지 제공 상황: chatType을 "chat"으로 설정
                final_response = {
                    "chatType": "chat",
                    "answer": answer_text,
                    "recipes": []
                }
            else:
                # 완전한 레시피 데이터: chatType을 "recipe"로 설정
                # 요리 제목 추출
                food_name = recipe_data.get("food_name") or recipe_data.get("title")
                
                # 답변 메시지 생성 - food_name이 있을 때만 레시피 메시지 사용
                if food_name:
                    answer_message = f"네, 요청하신 {food_name}의 레시피를 알려드릴게요."
                else:
                    # food_name이 없으면 원본 답변 그대로 사용
                    answer_message = recipe_data.get("answer", "")
                
                final_response = {
                    "chatType": "recipe",
                    "answer": answer_message,
                    "recipes": [recipe_data]
                }
            
            # JSON 문자열로 변환
            import json
            final_json = json.dumps(final_response, ensure_ascii=False, indent=2)
            
            # AIMessage 생성
            from langchain_core.messages import AIMessage
            final_response_msg = AIMessage(content=f"```json\n{final_json}\n```")
            
            logger.info(f"--- [LangGraph] ✍️ 최종 응답 (최적화 방식): {final_response_msg} ---")
            return {"messages": state["messages"] + [final_response_msg]}
        
        # 다건 병합 로직: 리스트 내 각 결과를 recipe/cart/chat 형태로 표준화 후 합치기
        if isinstance(parsed_output, list) and len(parsed_output) > 0:
            standardized_recipes = []
            chat_messages = []
            # 각 항목 표준화
            for entry in parsed_output:
                try:
                    if isinstance(entry, str):
                        # 이미 포매팅된 일반 문자열이면 스킵
                        continue
                    if isinstance(entry, dict):
                        # 비디오/텍스트 레시피 형태 표준화
                        source = entry.get("source") or ("video" if entry.get("video_info") else "text")
                        food_name = entry.get("food_name") or entry.get("title") or ""
                        ingredients = entry.get("ingredients") or entry.get("ingredients_raw") or []
                        steps = entry.get("recipe") or entry.get("steps") or []
                        # cart 결과(data.results)인 경우 product로 변환은 포매터에 맡기고 여기서는 recipe 빈 배열 유지
                        if "data" in entry and isinstance(entry.get("data"), dict) and "results" in entry["data"]:
                            # 장바구니 타입 후보: 후속 포매터로 전달하기 위해 그대로 append
                            standardized_recipes.append({
                                "source": "ingredient_search",
                                "food_name": entry.get("data", {}).get("query", "상품"),
                                "product": entry.get("data", {}).get("results", []),
                                "recipe": []
                            })
                        elif food_name or steps or ingredients:
                            standardized_recipes.append({
                                "source": source,
                                "food_name": food_name,
                                "ingredients": ingredients,
                                "recipe": steps
                            })
                        elif entry.get("answer"):
                            chat_messages.append(entry.get("answer"))
                except Exception:
                    continue

            # 우선순위: 레시피/카트가 하나라도 있으면 recipes 배열로 반환, 없으면 chat
            if standardized_recipes:
                # 요청 요리명 기반으로 안내 문구 생성
                recipe_names = []
                for r in standardized_recipes:
                    try:
                        name = (r.get("food_name") or "").strip()
                        has_recipe = isinstance(r.get("recipe"), list) and len(r.get("recipe") or []) > 0
                        has_ingredients = isinstance(r.get("ingredients"), list) and len(r.get("ingredients") or []) > 0
                        # cart 전용 항목(product만 있는 경우)은 제외하고, 레시피/재료가 있는 항목 위주로 이름 수집
                        if name and (has_recipe or has_ingredients):
                            recipe_names.append(name)
                    except Exception:
                        continue

                if len(recipe_names) == 1:
                    answer_message = f"요청하신 {recipe_names[0]}의 레시피를 알려드릴게요!"
                elif len(recipe_names) > 1:
                    joined = ", ".join(recipe_names)
                    answer_message = f"요청하신 {joined} 레시피를 알려드릴게요!"
                else:
                    answer_message = "요청하신 내용을 모두 정리해 드렸어요!"

                # 추가 권유: 원문에서 추출된 요리명 중 아직 처리되지 않은 것이 있으면 안내 문구 추가
                try:
                    from langchain_core.messages import HumanMessage as _HM
                    original_text = ""
                    for m in state.get("messages", []):
                        if isinstance(m, _HM):
                            original_text = str(m.content or "")
                            break
                    if original_text:
                        import re as _re
                        text_wo_urls = _re.sub(r"https?://\S+", " ", original_text)
                        # 구분자 통일: 와/과/랑/및/그리고/,+/ 등
                        normalized = _re.sub(r"\s*(와|과|랑|및|그리고|,|/|\+)\s*", ",", text_wo_urls)
                        # 잡어 제거
                        normalized = _re.sub(r"(레시피|조리법|만드는\s*법|알려줘|주세요|좀)", "", normalized)
                        parts = [p.strip() for p in normalized.split(",") if p.strip()]
                        # 남은 후보 계산 (대소문자 무시 비교)
                        lower_done = {n.lower() for n in recipe_names}
                        leftovers = [p for p in parts if p and p.lower() not in lower_done]
                        # 유튜브 링크만 남은 경우 제거 (안전차)
                        leftovers = [p for p in leftovers if not _re.search(r"youtube\.com|youtu\.be", p, _re.I)]
                        if len(leftovers) == 1:
                            answer_message += f" ‘{leftovers[0]}’ 레시피도 계속 보여드릴까요?"
                        elif len(leftovers) >= 2:
                            suggest = ", ".join(leftovers[:2])
                            answer_message += f" ‘{suggest}’ 중 어떤 걸 더 볼까요?"
                except Exception:
                    pass

                final_response = {
                    "chatType": "recipe",  # 혼합 시 기본은 recipe로 표기, cart 항목은 product 포함
                    "answer": answer_message,
                    "recipes": standardized_recipes
                }
            else:
                final_response = {
                    "chatType": "chat",
                    "answer": chat_messages[0] if chat_messages else "무엇을 도와드릴까요?",
                    "recipes": []
                }

            import json
            final_json = json.dumps(final_response, ensure_ascii=False, indent=2)
            from langchain_core.messages import AIMessage
            final_response_msg = AIMessage(content=f"```json\n{final_json}\n```")
            logger.info(f"--- [LangGraph] ✍️ 최종 응답 (다건 병합): {final_response_msg} ---")
            return {"messages": state["messages"] + [final_response_msg]}

        # 기존 방식 (fallback)
        final_response_msg = await formatter_chain.ainvoke(
            {"tool_output": tool_output}
        )
        logger.info(f"--- [LangGraph] ✍️ 최종 응답 (기존 방식): {final_response_msg} ---")
        return {"messages": state["messages"] + [final_response_msg]}
        
    except Exception as e:
        logger.error(f"--- [LangGraph] ✍️ 최종 응답 생성 중 오류: {e} ---")
        # 오류 발생시 기존 방식 사용
        final_response_msg = await formatter_chain.ainvoke(
            {"tool_output": state["tool_output"]}
        )
        return {"messages": state["messages"] + [final_response_msg]}


def should_call_tool(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "action"
    return END

# 4. 그래프(Graph) 생성 및 연결
workflow = StateGraph(AgentState)

# 1️⃣ 노드들을 먼저 그래프에 '등록'합니다.
workflow.add_node("agent", select_tool)
workflow.add_node("image_injector", inject_image_data)
workflow.add_node("action", custom_tool_node)
workflow.add_node("formatter", generate_final_answer)

# 2️⃣ 그래프의 시작점을 'agent' 노드로 설정합니다.
workflow.set_entry_point("agent")


# 3️⃣ 조건부 엣지 추가
workflow.add_conditional_edges(
    "agent",
    should_call_tool,
    {
        "action": "image_injector",
        END: "formatter",
    },
)

# 4️⃣ 엣지를 다시 연결합니다.
workflow.add_edge("image_injector", "action")
workflow.add_edge("action", "formatter")
workflow.add_edge("formatter", END)

# 5️⃣ 그래프를 컴파일합니다.
app = workflow.compile()
async def run_agent(input_data: dict):
    """
    사용자 입력을 받아 에이전트를 실행하고 결과를 반환합니다.
    input_data: {"message": str, "image_b64": Optional[str]}
    """
    try:
        user_message = input_data.get("message", "")
        image_bytes = input_data.get("image")

        inputs = {}

        if image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            if not user_message:
                user_message = "이 이미지에 있는 상품 정보를 찾아줘."

            full_message = f"{user_message} [사용자가 이미지를 첨부했습니다]"
            messages = [HumanMessage(content=full_message)]
            inputs = {
                "messages": messages,
                "image_b64": image_b64,
            }

        else:
            # 텍스트와 유튜브 링크를 동시에 허용: 링크는 보존, 텍스트도 그대로 전달
            # 단, 유튜브 링크가 여러 개면 마지막 링크 1개만 유지
            all_youtube_links = re.findall(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+', user_message)
            if len(all_youtube_links) > 1:
                latest_youtube_link = all_youtube_links[-1]
                # 최신 링크만 남기되, 원문 텍스트는 유지
                user_message = re.sub(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+', '', user_message)
                user_message = user_message.strip() + f"\n{latest_youtube_link}"

            messages = [
                HumanMessage(content=user_message or "")
            ]
            inputs = {
                "messages": messages,
                "image_b64": None,
            }

        if not messages:
            raise ValueError("처리할 메시지가 없습니다.")

        # LangGraph 실행 (내부 처리)
        result_state = await app.ainvoke(inputs)

        final_message = result_state["messages"][-1]
        output_string = (
            final_message.content if isinstance(final_message, AIMessage) else ""
        )

        if not output_string:
            return {"chatType": "chat", "answer": "무엇을 도와드릴까요?", "recipes": []}

        # 최종 결과에서 ```json ... ``` 부분을 추출
        clean_json_string = ""
        match = re.search(r"```(json)?\s*(\{.*?\})\s*```", output_string, re.DOTALL)

        if match:
            clean_json_string = match.group(2).strip()
        else:
            clean_json_string = output_string.strip()

        try:
            parsed_data = json.loads(clean_json_string)
        except Exception:
            parsed_data = {"chatType": "chat", "answer": "무엇을 도와드릴까요?", "recipes": []}

        return parsed_data

    except Exception as e:
        raise e
