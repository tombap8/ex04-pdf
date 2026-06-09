import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- UI 설정 ---
st.set_page_config(page_title="나만의 맞춤형 문서 요정", page_icon="🧚‍♂️", layout="wide")
st.title("🧚‍♂️ 나만의 맞춤형 문서 요정 (RAG Chatbot)")

# --- 함수 정의 ---
def process_uploaded_files(uploaded_files):
    """업로드된 파일들을 읽어 Document 객체 리스트로 반환합니다."""
    documents = []
    for uploaded_file in uploaded_files:
        # Streamlit의 UploadedFile은 임시 파일로 저장 후 처리해야 LangChain Loader와 호환성이 좋습니다.
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            elif uploaded_file.name.lower().endswith(".txt"):
                loader = TextLoader(temp_file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"{uploaded_file.name} 파일 처리 중 오류 발생: {e}")
        finally:
            # 처리 후 임시 파일 삭제
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
    return documents

def build_vector_store(documents, api_key):
    """문서들을 청킹하고 임베딩하여 Vector DB를 구축합니다."""
    # 1. 문서 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # 2. 임베딩 모델 및 벡터 저장소 생성 (FAISS)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

def create_rag_chain(vectorstore, api_key):
    """Vector DB를 기반으로 RAG 체인을 생성합니다."""
    # LLM 설정 (gpt-4o)
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0)
    
    # 시스템 프롬프트 설정 (페르소나 및 지침)
    system_prompt = (
        "당신은 친절하고 똑똑한 문서 요정입니다. "
        "제공된 검색된 문맥(context)을 사용하여 사용자의 질문에 답변하세요. "
        "문맥에 답변을 위한 정보가 없다면, 지어내지 말고 '제공된 문서에서는 해당 내용을 찾을 수 없습니다.'라고 답변하세요. "
        "답변은 최대한 정확하고 명확하게 작성하세요.\n\n"
        "문맥:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 문서 결합 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 검색기(Retriever) 설정: 상위 4개 문서 검색 (Similarity Search)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 최종 Retrieval 체인 생성
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- 상태 초기화 (Session State) ---
# Streamlit은 상호작용마다 코드가 재실행되므로, 유지되어야 할 상태를 session_state에 저장합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # 대화 기록 저장
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None # Vector DB 객체 저장
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None # 구성된 RAG 체인 저장

# --- 사이드바 ---
with st.sidebar:
    st.header("설정 ⚙️")
    
    # API Key 입력 (비밀번호 타입으로 마스킹 처리)
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("---")
    
    st.subheader("문서 업로드 📂")
    # 다중 파일 업로드 지원 (PDF, TXT)
    uploaded_files = st.file_uploader(
        "PDF 또는 TXT 파일을 업로드하세요.", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # 문서 학습 버튼
    if st.button("문서 학습 시작 🚀", use_container_width=True):
        if not api_key:
            st.warning("OpenAI API Key를 입력해주세요.")
        elif not uploaded_files:
            st.warning("업로드된 문서가 없습니다.")
        else:
            with st.spinner("문서를 분석하고 학습하는 중입니다... 요정이 열일 중! 🧚‍♂️✨"):
                # 1. 파일에서 텍스트 추출
                documents = process_uploaded_files(uploaded_files)
                
                if documents:
                    try:
                        # 2. Vector DB 구축 및 상태 저장
                        vectorstore = build_vector_store(documents, api_key)
                        st.session_state.vectorstore = vectorstore
                        
                        # 3. RAG 체인 생성 및 상태 저장
                        rag_chain = create_rag_chain(vectorstore, api_key)
                        st.session_state.rag_chain = rag_chain
                        
                        st.success("학습 완료! 이제 문서에 대해 질문해주세요.")
                    except Exception as e:
                        st.error(f"학습 중 오류가 발생했습니다: {e}")
                else:
                    st.error("문서에서 텍스트를 추출하지 못했습니다.")

# --- 메인 화면 (채팅 인터페이스) ---
# 1. 기존 대화 기록 출력
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 질문 입력창 및 처리 로직
# API 키가 없거나 문서 학습이 안된 경우, 입력창 대신 안내 메시지 출력
if not api_key:
    st.info("👈 사이드바에 OpenAI API Key를 입력해주세요.")
elif st.session_state.rag_chain is None:
    st.info("👈 사이드바에서 문서를 업로드하고 '문서 학습 시작' 버튼을 눌러주세요.")
else:
    # 사용자 질문 입력 활성화
    if prompt := st.chat_input("문서에 대해 무엇이든 물어보세요!"):
        # 사용자 질문 화면에 표시 및 세션 기록
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 챗봇 답변 생성 및 화면에 표시
        with st.chat_message("assistant"):
            with st.spinner("요정이 답변을 찾고 있어요... 🧚‍♂️🔍"):
                try:
                    # RAG 체인 실행 (사용자 질문 전달)
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # 답변 렌더링
                    st.markdown(answer)
                    
                    # 답변 세션 기록
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"답변 생성 중 오류가 발생했습니다. API Key나 할당량을 확인해주세요: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
