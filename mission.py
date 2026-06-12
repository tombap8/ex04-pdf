import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (  PyPDFLoader )
from langchain_text_splitters import (    RecursiveCharacterTextSplitter )
from langchain_chroma import (    Chroma  )
from langchain_openai import   OpenAIEmbeddings,    ChatOpenAI 
from langchain_classic.chains import (   create_retrieval_chain )
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import (   ChatPromptTemplate )

st.title("📄 PDF File Reader")
st.text("📌[API KEY 입력 필수]")
st.write("----------------")


openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "PDF 파일을 올려주세요",   type=["pdf"] )
st.write("----------------")

# 세션 상태 초기화 (대화 기록 및 체인 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

def pdf_to_document(uploaded_file):
    """    Streamlit 업로드 PDF를
    LangChain Document 형태로 변환
    """
    # 임시 폴더 생성
    temp_dir = tempfile.TemporaryDirectory()

    # 임시 PDF 파일
    temp_filepath = os.path.join(     temp_dir.name,    uploaded_file.name    )

    with open(   temp_filepath,    "wb"  ) as f:
        f.write(    uploaded_file.getvalue()    )

    loader = PyPDFLoader(   temp_filepath   )

    pages = loader.load()
    return pages

if uploaded_file is not None and openai_key:
    # 새로운 파일이 업로드되었을 때만 벡터 DB 생성 및 체인 초기화 수행 (최적화)
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("문서를 분석하고 벡터 DB를 구축 중입니다..."):
            pages = pdf_to_document(uploaded_file)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(pages)

            embeddings = OpenAIEmbeddings(api_key=openai_key)
            db = Chroma.from_documents(documents=texts, embedding=embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=openai_key,
                streaming=True # 스트리밍 활성화
            )

            prompt = ChatPromptTemplate.from_template(
                """
                당신은 PDF 분석 AI 입니다.
                Context:   {context}
                Question:  {input}
                답변:
                """
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            qa_chain = create_retrieval_chain(retriever, document_chain)

            # 생성된 체인과 파일명을 세션에 저장
            st.session_state.qa_chain = qa_chain
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = [] # 새 문서 로드 시 대화 내역 초기화
            
        st.success("✅ 문서 학습이 완료되었습니다! 아래에서 질문을 시작해보세요.")

# 체인이 구성되었으면 챗봇 UI 렌더링
if st.session_state.qa_chain is not None:
    st.header("💬 PDF에게 질문하세요")

    # 1. 이전 대화 기록 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 2. 채팅 입력창 (질문 입력 시 동작)
    if question := st.chat_input("문서 내용에 대해 질문하세요"):
        # 사용자 질문 화면 표시 및 세션 기록
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # AI 답변 스트리밍 생성 및 세션 기록
        with st.chat_message("assistant"):
            def stream_answer():
                for chunk in st.session_state.qa_chain.stream({"input": question}):
                    if answer_chunk := chunk.get("answer"):
                        yield answer_chunk

            # st.write_stream은 제너레이터의 결과를 실시간으로 출력하고 최종 조합된 문자열을 반환합니다.
            full_response = st.write_stream(stream_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_response})