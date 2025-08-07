# planning_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
from dotenv import load_dotenv


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
from shopping_service.agent import text_based_cooking_assistant
from video_service.core.extractor import extract_recipe_from_youtube

# 1. 사용할 도구(Tools) 정의
tools = [text_based_cooking_assistant, extract_recipe_from_youtube]

# 2. LLM 모델 설정 (Planning을 위해서는 고성능 모델을 추천합니다)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", 
    temperature=0, 
    convert_system_message_to_human=True,
    google_api_key=GEMINI_API_KEY,
)

# 3. 프롬프트(Prompt) 설정 - 에이전트에게 내리는 지시사항
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 사용자의 요청을 분석하여 최적의 도구를 사용해 답변하는 요리 전문 어시스턴트입니다.

    당신은 두 가지 도구를 사용할 수 있습니다:
    1. text_based_cooking_assistant: 텍스트로 된 요리 질문(레시피, 재료, 팁, 추천)에 답합니다.
    2. extract_recipe_from_youtube: 유튜브 URL에서 레시피를 추출합니다.

    ## 작동 규칙
    - 사용자의 질문에 유튜브 URL ('youtube.com' 또는 'youtu.be')이 포함되어 있으면 반드시 'extract_recipe_from_youtube' 도구를 사용하세요.
    - 사용자의 질문이 일반적인 텍스트 질문이라면 'text_based_cooking_assistant' 도구를 사용하세요.
    - 사용자의 요청이 복합적일 경우 (예: "이 유튜브 영상의 요리 이름이 뭐야? 그리고 그 요리 팁 좀 알려줘"),
      스스로 계획을 세워 도구들을 순서대로 사용하여 최종 답변을 만드세요.
      1. 먼저 'extract_recipe_from_youtube'로 영상의 핵심 정보를 추출합니다.
      2. 그 다음, 추출된 정보를 바탕으로 'text_based_cooking_assistant'에 추가 질문을 하여 답변을 완성합니다.
    - **extract_recipe_from_youtube 도구의 결과는 반드시 JSON 형태(food_name, ingredients, steps)를 그대로 반환하세요.**
    - 모든 답변은 친절한 한국어 말투로 정리해서 최종 사용자에게 전달해주세요.
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
    """에이전트를 실행하고 대화 기록을 관리하는 메인 함수"""
    # chat_history = chat_history_store.get(session_id, [])

    result = await agent_executor.ainvoke({
        "input": user_message,
        # "chat_history": chat_history
    })
    
    # 대화 기록 업데이트
    # chat_history.extend([
    #     HumanMessage(content=user_message),
    #     AIMessage(content=result["output"]),
    # ])
    # chat_history_store[session_id] = chat_history[-10:] # 최근 5개의 대화(질문+답변)만 저장

    return result["output"]
