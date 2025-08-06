# LangGraph, Gemini 분석 기능

import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import logging


# 다른 파일에 있는 스크립트 추출 함수를 가져옵니다.
from .transcript import get_youtube_transcript, get_youtube_title, get_youtube_duration

# --- 상태, 데이터 구조, 노드 등 LangGraph 관련 코드 ---

# Pydantic 모델 정의
class Recipe(BaseModel):
    food_name: str = Field(description="요리 이름")
    ingredients: List[str] = Field(description="요리에 필요한 재료 목록 (양 포함)")
    steps: List[str] = Field(description="조리 과정을 순서대로 요약한 목록")

class GraphState(TypedDict):
    youtube_url: str
    transcript: str
    video_title: str
    recipe: Recipe
    error: str


# 영상 제목 추출을 담당하는 노드
def title_node(state: GraphState) -> GraphState:
    logging.info("--- 영상 제목 추출 노드 실행 ---")
    try:
        video_title = get_youtube_title(state["youtube_url"])
        return {"video_title": video_title}
    except Exception as e:
        logging.error(f"영상 제목 추출 오류: {e}")
        return {"video_title": "요리명을 추출할 수 없습니다."}


# 스크립트 추출을 담당하는 노드
def transcript_node(state: GraphState) -> GraphState:
    logging.info("--- 스크립트 추출 노드 실행 ---")
    try:
        duration = get_youtube_duration(state["youtube_url"])
        logging.debug(f"DEBUG: 영상 길이(초): {duration}")
        if duration > 1200:
            logging.warning("WARN: 20분 초과 영상 - 처리 중단")
            return {"error": "20분을 초과하는 영상은 처리할 수 없습니다."}
        transcript_text = get_youtube_transcript(state["youtube_url"])
        logging.debug(f"DEBUG: 추출된 스크립트 길이: {len(transcript_text) if transcript_text else 0}")

        if not transcript_text or len(transcript_text.strip()) < 10:
            logging.warning("WARN: 스크립트가 없거나 너무 짧음")
            return {"error": "스크립트를 추출할 수 없습니다. (자막/음성 없음 또는 너무 짧음)"}
        logging.info(f"INFO: 스크립트 일부 미리보기: {transcript_text[:100]}...")
        return {"transcript": transcript_text}
    
    except Exception as e:
        logging.error(f"스크립트 추출 오류: {e}")
        return {"error": f"스크립트 추출 중 오류: {e}"}



# 영상 제목과 스크립트를 기반으로 레시피 영상인지 판단하는 노드
def recipe_validator_node(state: GraphState) -> GraphState:
    logging.info("--- AI 레시피 판별 노드 실행 ---")
    title = state.get("video_title", "")
    transcript = state.get("transcript", "")
    
    # 내용이 너무 짧으면 처리하지 않음
    if len(transcript) < 50:
        return {"error": "스크립트 내용이 너무 짧습니다."}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
        )
        
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
        logging.info(f"✅ AI 판별 결과: {result}")

        if "예" in result:
            return {} # 다음 단계로 진행 (에러 없음)
        else:
            return {"error": "AI가 레시피 영상이 아니라고 판단했습니다."}

    except Exception as e:
        logging.error(f"❌ AI 판별 중 오류: {e}")
        return {"error": f"AI 판별 중 오류 발생: {str(e)}"}


# 레시피 추출을 담당하는 노드
def recipe_extract_node(state: GraphState) -> GraphState:
    logging.info("--- 레시피 추출 노드 실행 ---")
    transcript = state.get("transcript")
    video_title = state.get("video_title", "요리명을 추출할 수 없습니다.")

    if not transcript:
        return {"error": "스크립트가 없습니다. (자막/음성 없음)"}

    # 이전에 있던 if not is_cooking_video블록을 완전히 삭제! 판별은 새로운 노드가 이미 수행했기 때문입니다.
    # if not is_cooking_video(video_title, transcript):
    #     return {"error": "이 영상은 요리 레시피 영상이 아니거나, 먹방/소개/리뷰 등 요리 과정이 포함되어 있지 않습니다."}

    try:
        # LLM 모델 초기화
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
        )

        # Pydantic 모델(Recipe)을 사용해 구조화된 출력을 요청
        structured_llm = llm.with_structured_output(Recipe)

        # 프롬프트 생성 - 더 구체적이고 명확한 지시사항
        prompt = f"""
        당신은 요리 레시피 전문가입니다. 주어진 유튜브 영상 제목과 스크립트를 바탕으로 레시피를 추출해주세요.

        - 스크립트에서 언급된 모든 재료를 ingredients 목록에 추가하세요.
        - 스크립트의 조리 과정을 순서대로 steps 목록에 추가하세요.

        [영상 제목]
        {video_title}

        [스크립트]
        {transcript}

        위 스크립트에서 요리의 재료와 조리 순서를 정확하게 추출해주세요.

        **중요한 지시사항:**
        1. 재료는 스크립트에서 언급된 모든 재료를 리스트로 정리해주세요.
        2. 조리 순서는 스크립트에서 실제로 언급된 모든 조리 단계를 순서대로 번호를 매겨 정리해주세요.
        3. 스크립트가 길면 더 많은 조리 단계가 있을 것입니다. 모든 단계를 놓치지 말고 추출해주세요.
        4. 각 조리 단계는 구체적이고 실용적인 내용으로 정리해주세요.
        5. 재료의 양이나 구체적인 수치가 언급되었다면 포함해주세요.
        """

        # LLM 호출
        recipe_object = structured_llm.invoke(prompt)
        logging.info(f"✅ LLM 구조화된 출력 결과: {recipe_object}")

        # Pydantic 객체를 state에 저장
        return {"recipe": recipe_object}
        
    except Exception as e:
        logging.error(f"레시피 추출 오류: {e}")
        return {"error": f"레시피 추출 중 오류 발생: {str(e)}"}


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


if __name__ == '__main__':
    # 그래프 객체 생성
    app = create_recipe_graph()

    # 그래프 시각화 코드 실행
    try:
        # get_graph()로 그래프 구조를 가져와 .draw_mermaid_png() 호출
        graph_image = app.get_graph().draw_mermaid_png()

        # 이미지 파일로 저장
        with open("graph_mermaid.png", "wb") as f:
            f.write(graph_image)
        print("✅ Mermaid 그래프 이미지가 'graph_mermaid.png' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"❌ Mermaid 그래프 이미지 생성 실패: {e}")
        print("playwright가 제대로 설치되었는지 확인해주세요. (playwright install 실행)")

    # 테스트 실행
    test_url = "https://www.youtube.com/watch?v=qWbHSOplcvY"  # 테스트할 URL
    result = app.invoke({"youtube_url": test_url})
    print("\n--- 최종 결과 ---")
    print(result)

