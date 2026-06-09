import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import requests
import re

# 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 로드)
load_dotenv()

st.set_page_config(page_title="유튜브 요약 AI", page_icon="🎥", layout="centered")

st.title("🎥 유튜브 요약 AI")
st.markdown("""
유튜브 영상 링크를 입력하면 AI가 영상의 자막을 추출하여 내용을 깔끔하게 요약해 줍니다. 
**해외 영상도 한국어로 자동 번역되어 요약됩니다!**
*(주의: 작동을 위해서는 `.env` 파일에 `OPENAI_API_KEY`가 설정되어 있어야 합니다.)*
""")

def extract_video_id(url):
    """유튜브 URL에서 비디오 ID를 추출합니다."""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            p = parse_qs(parsed_url.query)
            return p.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    return None

def get_youtube_title(video_id):
    """유튜브 영상 제목을 HTML에서 추출합니다."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url, timeout=5)
        match = re.search(r'<title>(.*?)</title>', response.text)
        if match:
            return match.group(1).replace(" - YouTube", "").strip()
    except:
        pass
    return "제목을 불러올 수 없는 영상"

def get_transcript(video_id):
    """비디오 ID로 자막을 가져옵니다. 한국어가 없으면 다른 언어를 가져옵니다."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        # 1. 한국어 자막이 있는지 우선 확인
        try:
            transcript = transcript_list.find_transcript(['ko'])
        except Exception:
            # 2. 없다면 사용 가능한 아무 자막이나 가져오기 (이후 LLM이 알아서 번역)
            transcript = next(iter(transcript_list))
            
        fetched_transcript = transcript.fetch()
        text = " ".join([t.text for t in fetched_transcript])
        
        # 텍스트가 너무 길 경우 토큰 제한 및 비용 절감을 위해 자르기 (약 3~4시간 분량)
        max_length = 100000
        if len(text) > max_length:
            text = text[:max_length] + "... [텍스트가 너무 길어 후반부가 생략되었습니다]"
            
        return text
    except Exception as e:
        st.error(f"자막을 가져오는데 실패했습니다. 자막이 제공되지 않는 영상일 수 있습니다. (에러: {e})")
        return None

def summarize_text(text):
    """LangChain을 사용하여 텍스트를 요약하고 한국어로 작성합니다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하거나 환경 변수를 설정해주세요.")
        
    # 모델 설정 (비용 효율적이고 성능이 좋은 gpt-4o-mini 사용)
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    
    prompt = PromptTemplate.from_template(
        "다음은 유튜브 영상의 자막입니다.\n"
        "이 내용을 바탕으로 전체 내용을 요약해 주세요.\n"
        "원문이 어떤 언어이든 상관없이 **반드시 한국어로 자연스럽게** 작성해 주세요.\n\n"
        "자막 내용:\n{text}\n\n"
        "요약 결과:"
    )
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content

# UI: 링크 입력 창
url_input = st.text_input("유튜브 링크 주소를 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

# UI: 요약 버튼
if st.button("요약하기", type="primary"):
    if url_input:
        video_id = extract_video_id(url_input)
        if video_id:
            with st.spinner("영상 정보와 자막을 추출하고 있습니다..."):
                title = get_youtube_title(video_id)
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                transcript_text = get_transcript(video_id)
            
            if transcript_text:
                with st.spinner("AI가 내용을 요약 중입니다... 잠시만 기다려주세요."):
                    try:
                        summary = summarize_text(transcript_text)
                        
                        st.divider()
                        
                        # --- 결과 출력 UI 디자인 ---
                        # 1. 영상 정보 영역 (썸네일 + 제목)
                        st.markdown("### 🎬 영상 정보")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(thumbnail_url, use_container_width=True)
                        with col2:
                            st.subheader(title)
                            st.markdown(f"**유튜브 링크:** [바로가기]({url_input})")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # 2. 요약 내용 영역 (디자인 박스 사용)
                        st.markdown("### 💡 AI 요약 내용")
                        st.info(summary, icon="📝")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # 3. 원본 자막 토글 영역
                        with st.expander("🔍 원본 자막 텍스트 보기"):
                            st.write(transcript_text)
                            
                    except Exception as e:
                        st.error(f"요약 중 오류가 발생했습니다. ({e})")
        else:
            st.warning("유효한 유튜브 링크가 아닙니다. 링크를 다시 확인해 주세요.")
    else:
        st.warning("먼저 유튜브 링크를 입력해 주세요.")
