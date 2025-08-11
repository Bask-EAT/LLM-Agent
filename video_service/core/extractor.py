# LangGraph, Gemini 분석 기능

import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import logging
from config import GEMINI_API_KEY


# 다른 파일에 있는 스크립트 추출 함수를 가져옵니다.
from .transcript import get_youtube_transcript, get_youtube_title, get_youtube_duration

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic 모델 정의
class Ingredient(BaseModel):
    item: str = Field(description="재료 이름")
    amount: str = Field(description="재료의 양")
    unit: str = Field(description="재료의 단위 (예: g, ml, 큰술, 개)")

class Recipe(BaseModel):
    food_name: str = Field(description="요리 이름")
    ingredients: List[Ingredient] = Field(description="요리에 필요한 재료 목록 (양 포함)")
    recipe: List[str] = Field(description="조리 과정을 순서대로 요약한 목록. 'steps'가 아닌 'recipe' 키를 사용해야 합니다.")

class GraphState(TypedDict):
    youtube_url: str
    transcript: str
    video_title: str
    recipe: Recipe
    error: str
    final_answer: str


# 영상 제목 추출을 담당하는 노드
def title_node(state: GraphState) -> GraphState:
    logger.info("--- 영상 제목 추출 노드 실행 ---")
    try:
        video_title = get_youtube_title(state["youtube_url"])
        return {"video_title": video_title}
    except Exception as e:
        logger.error(f"영상 제목 추출 오류: {e}")
        return {"video_title": "요리명을 추출할 수 없습니다."}


# 스크립트 추출을 담당하는 노드
def transcript_node(state: GraphState) -> GraphState:
    logger.info("--- 스크립트 추출 노드 실행 ---")
    try:
        duration = get_youtube_duration(state["youtube_url"])
        logger.debug(f"DEBUG: 영상 길이(초): {duration}")
        if duration > 1200:
            logger.warning("WARN: 20분 초과 영상 - 처리 중단")
            return {"error": "20분을 초과하는 영상은 처리할 수 없습니다."}
        transcript_text = get_youtube_transcript(state["youtube_url"])
        logger.debug(f"DEBUG: 추출된 스크립트 길이: {len(transcript_text) if transcript_text else 0}")

        if not transcript_text or len(transcript_text.strip()) < 10:
            logger.warning("WARN: 스크립트가 없거나 너무 짧음")
            return {"error": "스크립트를 추출할 수 없습니다. (자막/음성 없음 또는 너무 짧음)"}
        logger.info(f"INFO: 스크립트 일부 미리보기: {transcript_text[:100]}...")
        return {"transcript": transcript_text}
    
    except Exception as e:
        logger.error(f"스크립트 추출 오류: {e}")
        return {"error": f"스크립트 추출 중 오류: {e}"}



# 영상 제목과 스크립트를 기반으로 레시피 영상인지 판단하는 노드
def recipe_validator_node(state: GraphState) -> GraphState:
    logger.info("--- AI 레시피 판별 노드 실행 ---")
    title = state.get("video_title", "")
    transcript = state.get("transcript", "")
    
    # 내용이 너무 짧으면 처리하지 않음
    if len(transcript) < 50:
        return {"error": "스크립트 내용이 너무 짧습니다."}

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        
        prompt = f"""
        주어진 영상 제목과 스크립트를 보고, 이 영상이 음식을 만들거나 조리하는 방법에 대한 정보를 포함하고 있는지 판단해줘.

        - 단순히 음식을 먹기만 하는 '먹방'이나, 식당을 '리뷰' 또는 '소개'하는 영상은 '아니오'로 판단해야 해.
        - 라면을 끓이거나, 기존 제품을 섞어 먹는 등 아주 간단한 조리법이라도 포함되어 있다면 '예'로 판단해야 해.

        [영상 제목]
        {title}

        [스크립트]
        {transcript[:1000]}  # 스크립트가 너무 길 경우를 대비해 앞부분만 사용

        위 내용을 바탕으로 판단했을 때, 레시피 정보가 포함되어 있다면 '예', 그렇지 않다면 '아니오' 둘 중 하나로만 대답해줘.
        """
        
        result = llm.invoke(prompt).content.strip()
        logger.info(f"✅ AI 판별 결과: {result}")

        if "예" in result:
            return {} # 다음 단계로 진행 (에러 없음)
        else:
            return {"error": "AI가 레시피 영상이 아니라고 판단했습니다."}

    except Exception as e:
        logger.error(f"❌ AI 판별 중 오류: {e}")
        return {"error": f"AI 판별 중 오류 발생: {str(e)}"}


# 레시피 추출을 담당하는 노드
def recipe_extract_node(state: GraphState) -> GraphState:
    logger.info("--- 레시피 추출 노드 실행 ---")
    transcript = state.get("transcript")
    video_title = state.get("video_title", "요리명을 추출할 수 없습니다.")

    if not transcript:
        return {"error": "스크립트가 없습니다. (자막/음성 없음)"}

    try:
        # LLM 모델 초기화
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)

        # Pydantic 모델(Recipe)을 사용해 구조화된 출력을 요청
        structured_llm = llm.with_structured_output(Recipe)

        # 프롬프트 생성 - 더 구체적이고 명확한 지시사항
        prompt = f"""
        당신은 요리 레시피 전문가입니다. 주어진 유튜브 영상 제목과 스크립트를 바탕으로 레시피를 추출해주세요.

        - 스크립트에서 언급된 모든 재료를 ingredients 목록에 추가하세요.
        - 스크립트의 조리 과정을 순서대로 recipe 목록에 추가하세요.

        [영상 제목]
        {video_title}

        [스크립트]
        {transcript}

        위 스크립트에서 요리의 재료와 조리 순서를 정확하게 추출해주세요.

        **중요한 지시사항:**
        1. 재료는 스크립트에서 언급된 모든 재료를 리스트로 정리해주세요. 각 객체는 **item**, **amount**, **unit** 키를 포함해야 합니다.
        2. 조리 순서는 스크립트에서 실제로 언급된 모든 조리 단계를 순서대로 번호를 매겨 정리해주세요.
        3. 조리 순서는 **최대 15단계**까지만 생성해주세요.
        4. 각 조리 단계는 구체적이고 실용적인 내용으로 정리해주세요.
        5. 재료의 양이나 구체적인 수치가 언급되었다면 반드시 amount와 unit에 포함해주세요. 양이 불명확하면 '적당량', '약간' 등으로 표시하세요.
        """

        # LLM 호출
        recipe_object = structured_llm.invoke(prompt)
        logger.info(f"✅ LLM 구조화된 출력 결과: {recipe_object}")

        # 사용자에게 보여줄 최종 답변을 생성합니다.
        ingredient_list_str = [
            f"{ing.item} {ing.amount} {ing.unit}".strip()
            for ing in recipe_object.ingredients
        ]
        answer = (f"✅ 유튜브 영상에서 '{recipe_object.food_name}' 레시피를 성공적으로 추출했습니다!\n\n"
                  f"📋 **필요한 재료:**\n- " + "\n- ".join(ingredient_list_str) + "\n\n"
                  f"👨‍🍳 **조리법:**\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe_object.recipe)))

        # Pydantic 객체를 state에 저장
        return {"recipe": recipe_object, "final_answer": answer}
        
    except Exception as e:
        logger.error(f"레시피 추출 오류: {e}")
        return {"error": f"레시피 추출 중 오류 발생: {str(e)}"}



def should_continue(state: GraphState) -> str:
    # 에러가 있으면 그래프를 종료하고, 없으면 다음 단계로 진행합니다.
    return END if state.get("error") else "extractor"


# --- 그래프 구성 ---
def create_recipe_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("title_extractor", title_node)
    workflow.add_node("transcriber", transcript_node)
    workflow.add_node("validator", recipe_validator_node)  # 판별 노드 추가
    workflow.add_node("extractor", recipe_extract_node)
    
    workflow.set_entry_point("title_extractor")
    workflow.add_edge("title_extractor", "transcriber")
    workflow.add_edge("transcriber", "validator") # transcriber 다음에 validator 실행

    # validator 결과에 따라 분기 처리
    def should_continue(state: GraphState) -> str:
        return "extractor" if not state.get("error") else END

    workflow.add_conditional_edges("validator", should_continue, {
        "extractor": "extractor",
        END: END
    })
    workflow.add_edge("extractor", END)
    
    return workflow.compile()


# FastAPI 서비스용 함수
def process_video_url(youtube_url: str) -> dict:
    """FastAPI에서 호출할 메인 함수"""
    try:
        # 그래프 객체 생성
        app = create_recipe_graph()
        
        # 그래프 실행
        result = app.invoke({"youtube_url": youtube_url})
        
        # 결과 처리
        if "error" in result:
            return {
                "answer": f"영상 처리 중 오류가 발생했습니다: {result['error']}",
                "food_name": recipe.food_name,
                "ingredients": [],
                "recipe": []
            }
        
        if "recipe" in result:
            recipe = result["recipe"]
            return {
                "answer": f"✅ {recipe.food_name} 레시피를 성공적으로 추출했습니다!\n\n📋 재료: {len(recipe.ingredients)}개\n👨‍🍳 조리 단계: {len(recipe.recipe)}단계",
                "food_name": recipe.food_name,
                "ingredients": recipe.ingredients,
                "recipe": recipe.recipe
            }
        
        return {
            "answer": "레시피를 추출할 수 없습니다.",
            "food_name": recipe.food_name,
            "ingredients": [],
            "recipe": []
        }
        
    except Exception as e:
        print(f"VideoAgent 처리 오류: {e}")
        return {
            "answer": f"영상 처리 중 오류가 발생했습니다: {str(e)}",
            "ingredients": [],
            "recipe": []
        } 
    

# --- LangChain 도구(Tool) 정의 ---
@tool
def extract_recipe_from_youtube(youtube_url: str) -> str:
    """
    유튜브(YouTube) URL에서 요리 레시피(재료, 조리법)를 추출할 때 사용합니다.
    사용자가 유튜브 링크를 제공하며 레시피를 분석, 요약, 또는 추출해달라고 요청할 경우에만 이 도구를 사용해야 합니다.
    입력값은 반드시 유튜브 URL이어야 합니다.
    """
    logger.info(f"유튜브 레시피 추출 도구 실행: {youtube_url}")
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return "유효한 유튜브 URL이 아닙니다. 다시 확인해주세요."
        
    app = create_recipe_graph()
    # LangGraph 실행
    result = app.invoke({"youtube_url": youtube_url})
    
    # 최종 결과 또는 에러 메시지를 반환합니다.
    if result.get("error"):
        return result.get("error")
    
    #  Pydantic 객체를 JSON 문자열로 반환
    if result.get("recipe"):
        return result["recipe"].model_dump_json()
    
    return "알 수 없는 오류가 발생했습니다."