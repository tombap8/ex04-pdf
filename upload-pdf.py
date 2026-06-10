import streamlit as st
import tempfile
import os
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

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# PDF 메타데이터에서 소설 정보 추출하는 함수
def get_pdf_metadata(uploaded_file):
    """업로드된 PDF 파일 객체에서 제목과 작가 정보 추출"""
    try:
        # 파일 포인터를 처음으로 되돌림
        uploaded_file.seek(0)
        pdf_reader = PdfReader(uploaded_file)
        metadata = pdf_reader.metadata
        
        title = None
        author = None
        
        # 메타데이터에서 제목과 작가 찾기
        if metadata:
            if "/Title" in metadata:
                title_val = metadata["/Title"]
                if title_val:
                    title = str(title_val).strip()
            
            if "/Author" in metadata:
                author_val = metadata["/Author"]
                if author_val and str(author_val).strip():
                    author = str(author_val).strip()
        
        # 메타데이터에 없으면 PDF 첫 페이지의 첫, 두 번째 줄에서 추출
        if not title or not author:
            try:
                first_page_text = pdf_reader.pages[0].extract_text()
                if first_page_text:
                    lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
                    
                    # 첫 줄을 제목으로 사용 (메타데이터에 없는 경우)
                    if not title and lines:
                        title = uploaded_file.name.replace(".pdf", "")
                        # title = lines[0]
                    
                    # 두 번째 줄을 작가로 사용 (메타데이터에 없는 경우)
                    if not author and len(lines) > 1:
                        author = lines[1]
            except Exception as e:
                pass
        
        # 기본값 설정
        if not title:
            title = uploaded_file.name.replace(".pdf", "")
        if not author:
            author = "알 수 없는 작가"
            
        return title.strip(), author.strip()
    except Exception as e:
        return uploaded_file.name.replace(".pdf", ""), "알 수 없는 작가"


def process_pdf_and_create_chain(uploaded_file):
    """업로드된 PDF 파일을 가공하여 RAG 체인 생성"""
    # 1. 업로드된 파일을 임시 파일로 저장 (PyPDFLoader 사용을 위함)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 2. PDF 파일 로드 및 분할
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
    finally:
        # 파일 처리가 끝나면 임시 파일 삭제
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    # 3. 텍스트를 청크 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,           # 하나의 청크가 가질 최대 글자 수
        chunk_overlap=20,         # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # 4. OpenAI 텍스트 임베딩 수행
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. 벡터 데이터베이스 구축
    db = FAISS.from_documents(texts, embeddings_model)

    # 6. GPT 모델 및 MultiQueryRetriever 구성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), 
        llm=llm
    )

    # 7. 시스템 프롬프트 정의
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    
    # 8. 대화 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 9. 체인 통합
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain


# 메인 앱 타이틀
st.title("📚 개인 맞춤형 RAG 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "novel_title" not in st.session_state:
    st.session_state.novel_title = ""
if "novel_author" not in st.session_state:
    st.session_state.novel_author = ""

# 사이드바 영역: PDF 파일 업로드 및 가공 진행
with st.sidebar:
    st.header("📁 PDF 파일 업로드")
    uploaded_file = st.file_uploader(
        "사용할 PDF 문서를 첨부하세요.", 
        type=["pdf"],
        help="소설, 기사, 업무 문서 등 질문하고 싶은 PDF 파일을 올려주세요."
    )
    
    # 새로운 파일이 업로드된 경우 가공 프로세스 실행
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("🔄 문서를 분석하고 인덱싱하는 중입니다..."):
                try:
                    # 메타데이터 추출
                    title, author = get_pdf_metadata(uploaded_file)
                    st.session_state.novel_title = title
                    st.session_state.novel_author = author
                    
                    # RAG 체인 빌드 및 세션 저장
                    st.session_state.rag_chain = process_pdf_and_create_chain(uploaded_file)
                    st.session_state.current_file = uploaded_file.name
                    
                    # 새로운 파일 로드 시 이전 대화 초기화
                    st.session_state.messages = []
                    st.success("✅ 문서 분석 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류 발생: {str(e)}")

# 메인 콘텐츠 영역 분기 처리
if st.session_state.rag_chain is not None:
    # 업로드 완료 시 헤더 노출
    st.header(f"📖 {st.session_state.novel_title}")
    st.write(f"**작가/출처:** ✍️ {st.session_state.novel_author}")
    st.divider()

    # 2단 레이아웃 분할
    col_left, col_right = st.columns(2)

    # 왼쪽 레이아웃: 질문 입력 폼
    with col_left:
        st.subheader("❓ 질문 입력")
        
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "문서에 대해 알고 싶은 내용을 질문하세요:",
                placeholder="예: 등장 인물의 관계를 설명해줘."
            )
            search_button = st.form_submit_button("🔍 검색", use_container_width=True)
        
        # 대화 기록 초기화 버튼
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 오른쪽 레이아웃: 메세지 히스토리 출력
    with col_right:
        st.subheader("💬 대화 기록")
        
        if st.session_state.messages:
            messages = st.session_state.messages
            pairs = []
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    pairs.append((messages[i], messages[i+1]))
            
            # 최신 대화 기록이 위로 배치되도록 역순 정렬
            for user_msg, assistant_msg in reversed(pairs):
                st.chat_message("user").write(f"**질문:** {user_msg['content']}")
                st.chat_message("assistant").write(f"**답변:** {assistant_msg['content']}")
                if assistant_msg.get("context_count"):
                    st.caption(f"📖 검색된 참조 문서 청크: {assistant_msg['context_count']}개")
        else:
            st.info("아직 대화가 없습니다. 왼쪽에서 질문을 입력해 주세요!")

    # 사용자가 질문을 입력하고 폼을 제출했을 때 동작
    if search_button and question:
        with st.spinner("⏳ 답변을 구성하고 있습니다..."):
            try:
                # 저장된 RAG 체인을 가져와 질문 수행
                response = st.session_state.rag_chain.invoke({"input": question})
                
                # 대화 히스토리에 기록 저장
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "context_count": len(response.get('context', []))
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ 답변 생성 오류: {str(e)}")
else:
    # 파일을 아직 올리지 않았을 때 노출하는 가이드 화면
    st.info("👈 왼쪽 사이드바에서 PDF 문서를 먼저 업로드해 주세요.")
    st.markdown("""
    ### 💡 사용 방법 안내
    1. 왼쪽 사이드바의 **'PDF 파일 업로드'** 영역에 질문하고자 하는 문서를 끌어다 놓거나 선택함.
    2. 시스템이 문서를 자동으로 파싱하고 의미 단위(Chunk)로 쪼개어 임베딩 DB를 구성함.
    3. 완료 메시지가 표시되면 오른쪽 화면에서 자유롭게 문서 기반 질문-답변을 수행함.
    """)