import streamlit as st
import tempfile
import os
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 페이지 설정 (기획서와 프롬프트를 넓게 보기 위해 wide 모드 유지)
st.set_page_config(
    page_title="AI 뮤직비디오 스토리보드 디렉터",
    page_icon="🎬",
    layout="wide"
)

def get_file_metadata(uploaded_file):
    """파일 타입에 따라 제목 추출 (PDF/TXT 공통)"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    title = uploaded_file.name.replace(f".{file_ext}", "")
    
    if file_ext == "pdf":
        try:
            uploaded_file.seek(0)
            pdf_reader = PdfReader(uploaded_file)
            metadata = pdf_reader.metadata
            if metadata and "/Title" in metadata and metadata["/Title"]:
                title = str(metadata["/Title"]).strip()
        except:
            pass
    return title, "사용자 업로드 소스"

def process_file_and_create_chain(uploaded_file):
    """PDF 또는 TXT 파일을 가공하여 RAG 체인 생성"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    texts = []
    
    # 텍스트 분할기 설정 (가사나 시나리오는 문맥 유지가 중요하므로 청크 사이즈 조절)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        length_function=len,
    )

    if file_ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
            texts = text_splitter.split_documents(pages)
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                
    elif file_ext in ["txt", "md"]:
        # 텍스트 파일 바로 읽기
        raw_text = str(uploaded_file.read(), "utf-8")
        # 랑체인 다큐먼트 형식으로 변환 후 분할
        from langchain_core.documents import Document
        doc = [Document(page_content=raw_text, metadata={"source": uploaded_file.name})]
        texts = text_splitter.split_documents(doc)

    # 벡터 DB 및 리트리버 빌드
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(texts, embeddings_model)
    
    # 가성비가 좋고 빠른 모델 설정 (여기서는 기본 gpt-4o-mini 유지, xAI 연동 시 수정 가능)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) # 창의성을 위해 temperature를 0.5로 약간 올림
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)

    # 프롬프트 및 스토리보드 생성에 특화된 시스템 프롬프트 변경
    system_prompt = (
        "너는 전 세계를 매료시킨 천재 뮤직비디오 감독이자 최고 수준의 AI 프롬프트 엔지니어 가이드야.\n"
        "아래 제공된 가사 및 기획서 맥락(context)을 바탕으로 음악의 감정선, 비트, 세계관에 딱 맞는 독창적인 뮤직비디오 스토리보드를 설계해줘.\n\n"
        "지시사항:\n"
        "1. 질문자가 요청한 가사 구간이나 분위기에 맞춰 컷별 연출안을 작성할 것.\n"
        "2. 출력은 반드시 다음 구조의 Markdown 표(Table) 형식으로만 채워서 반환할 것. 다른 부연 설명은 하지 마.\n"
        "| 컷 번호 | 매핑 가사/파트 | 비주얼 연출 콘셉트 (한글 묘사) | 이미지 프롬프트 (영어 Midjourney 스타일) | 영상 프롬프트 (영어 Runway/Sora 스타일) |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n\n"
        "맥락:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_from_llm, question_answer_chain)


# --- UI 구현부 ---
st.title("🎬 AI 뮤직비디오 스토리보드 디렉터")
st.subheader("📌 가사나 기획서(PDF, TXT)를 올리면 AI 연출안과 프롬프트를 자동으로 뽑아줍니다.")

if "current_file" not in st.session_state: st.session_state.current_file = None
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "doc_title" not in st.session_state: st.session_state.doc_title = ""
if "generated_storyboard" not in st.session_state: st.session_state.generated_storyboard = None

# 사이드바: 파일 업로드 (PDF, TXT, MD 지원)
with st.sidebar:
    st.header("📁 소스 파일 업로드")
    uploaded_file = st.file_uploader(
        "가사 파일 또는 세계관 기획서를 첨부하세요.", 
        type=["pdf", "txt", "md"],
        help="소설, 노래 가사, 콘셉트 기획서 등을 지원합니다."
    )
    
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("🔄 텍스트 분석 및 세계관 인덱싱 중..."):
                try:
                    title, author = get_file_metadata(uploaded_file)
                    st.session_state.doc_title = title
                    st.session_state.rag_chain = process_file_and_create_chain(uploaded_file)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.generated_storyboard = None # 새 파일 로드 시 결과 초기화
                    st.success("✅ 분석 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")

# 메인 콘텐츠 영역 분기
if st.session_state.rag_chain is not None:
    st.header(f"📖 분석 중인 소스: {st.session_state.doc_title}")
    st.divider()

    col_left, col_right = st.columns([1, 2]) # 결과창을 더 넓게 쓰기 위해 비율 조정 [1:2]

    # 왼쪽 레이아웃: 디렉팅 명령 입력
    with col_left:
        st.subheader("🤖 연출 지시하기")
        with st.form("generation_form"):
            user_instruction = st.text_area(
                "AI 감독에게 내릴 요구사항을 적으세요:",
                value="전체 가사를 바탕으로 시네마틱하고 몽환적인 사이버펑크 스타일의 5개 핵심 컷 스토리보드를 짜줘. AI 걸그룹 멤버들이 주인공이야.",
                height=150
            )
            submit_button = st.form_submit_button("🎬 스토리보드 및 프롬프트 생성", use_container_width=True)

    # 오른쪽 레이아웃: 변경된 결과 창 (스토리보드 전용 뷰어)
    with col_right:
        st.subheader("📋 생성된 스토리보드 결과")
        
        if submit_button and user_instruction:
            with st.spinner("⏳ AI 감독이 컷을 구성하고 프롬프트를 엔지니어링 중입니다..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_instruction})
                    # 결과를 세션에 저장하여 고정
                    st.session_state.generated_storyboard = response['answer']
                except Exception as e:
                    st.error(f"❌ 생성 오류: {str(e)}")
        
        # 결과 출력 마크다운 뷰어
        if st.session_state.generated_storyboard:
            st.markdown(st.session_state.generated_storyboard)
            st.caption("💡 위 테이블의 영어 프롬프트를 복사하여 Midjourney나 Runway, Grok 등에 그대로 붙여넣어 사용하세요.")
        else:
            st.info("왼쪽에서 연출 지시문 입력 후 버튼을 누르면 여기에 표 형태로 스토리보드가 출력됩니다.")
else:
    st.info("👈 왼쪽 사이드바에서 가사가 담긴 PDF 또는 TXT 파일을 먼저 업로드해 주세요.")