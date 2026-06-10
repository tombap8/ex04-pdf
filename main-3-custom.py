import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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

# PDF 메타데이터에서 소설 정보 추출
@st.cache_resource
def get_pdf_metadata():
    """PDF 파일에서 제목과 작가 정보 추출"""
    try:
        pdf_reader = PdfReader("novel.pdf")
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
                        title = lines[0]
                        print(f"📖 첫 페이지에서 제목 추출: {title}")
                    
                    # 두 번째 줄을 작가로 사용 (메타데이터에 없는 경우)
                    if not author and len(lines) > 1:
                        author = lines[1]
                        print(f"✍️ 첫 페이지에서 작가 추출: {author}")
            except Exception as e:
                print(f"❌ PDF 텍스트 추출 실패: {e}")
        
        # 기본값 설정
        if not title:
            title = "알 수 없는 제목"
        if not author:
            author = "알 수 없는 작가"
            
        return title.strip(), author.strip()
    except Exception as e:
        print(f"❌ PDF 메타데이터 읽기 오류: {str(e)}")
        return "알 수 없는 제목", "알 수 없는 작가"

st.title("📚 RAG 챗봇 - PDF 질문 답변")

# 캐시를 사용하여 RAG 체인 초기화 (한 번만 실행)
@st.cache_resource
def load_rag_chain():
    """RAG 체인 로드 및 초기화"""
    # PDF 파일을 로드하고 페이지 단위로 분할
    loader = PyPDFLoader("novel.pdf")
    pages = loader.load_and_split()

    # 텍스트를 작은 청크 단위로 분할하여 의미있는 정보 단위로 재구성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
        chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
        is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(pages)

    # OpenAI의 텍스트 임베딩 모델을 사용하여 텍스트를 벡터로 변환
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터화된 청크들을 Chroma 벡터 데이터베이스에 저장
    db = Chroma.from_documents(texts, embeddings_model)

    # GPT-4o-mini 모델을 사용하여 질문-답변 작업 수행
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 사용자의 질문으로부터 여러 변형된 질문을 생성하여 검색 정확도 향상
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), 
        llm=llm
    )

    # RAG 챗봇의 역할과 동작 방식을 정의하는 시스템 프롬프트
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    
    # 시스템 프롬프트와 사용자 입력을 조합하여 채팅 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 검색된 문서들을 LLM에 함께 전달하여 질문에 대한 답변을 생성
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 검색기와 질문-답변 체인을 결합하여 RAG 파이프라인 구성
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain

# RAG 체인 로드 (PDF 처리)
try:
    with st.spinner("🔄 RAG 모델을 로딩 중입니다..."):
        rag_chain = load_rag_chain()
    st.success("✅ 모델 로딩 완료!")
except Exception as e:
    st.error(f"❌ 모델 로딩 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# RAG 로드 완료 후 PDF 메타데이터 가져오기 및 표시
NOVEL_TITLE, NOVEL_AUTHOR = get_pdf_metadata()

st.header(f"📖 {NOVEL_TITLE}")
st.write(f"**작가:** ✍️ {NOVEL_AUTHOR}")

st.divider()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 메인 컨텐츠를 2개 컬럼으로 분할
col_left, col_right = st.columns(2)

# 왼쪽: 질문 입력
with col_left:
    st.subheader("❓ 질문 입력")
    
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 아내가 먹고 싶어하는 음식은 무엇이야?"
        )
        
        search_button = st.form_submit_button("🔍 검색", use_container_width=True)
    
    # 초기화 버튼
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 오른쪽: 대화 기록
with col_right:
    st.subheader("💬 대화 기록")
    
    # 대화 기록 표시 (최신 질문-답변 쌍이 맨 위)
    if st.session_state.messages:
        # 메시지를 2개씩 묶어서 (질문, 답변) 쌍으로 변환
        messages = st.session_state.messages
        pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                pairs.append((messages[i], messages[i+1]))
        
        # 역순으로 표시 (최신이 맨 위)
        for user_msg, assistant_msg in reversed(pairs):
            st.chat_message("user").write(f"**질문:** {user_msg['content']}")
            st.chat_message("assistant").write(f"**답변:** {assistant_msg['content']}")
            if assistant_msg.get("context_count"):
                st.caption(f"📖 검색된 참조 문서: {assistant_msg['context_count']}개")
    else:
        st.info("아직 대화가 없습니다. 왼쪽에서 질문해주세요!")

# 질문 처리
if search_button and question:
    with st.spinner("⏳ 답변을 생성 중입니다..."):
        try:
            response = rag_chain.invoke({"input": question})
            
            # 메시지 히스토리에 추가
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
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
