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
from langchain_core.callbacks import   BaseCallbackHandler

st.title("📄 PDF File Reader")
st.write("----------------")


openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "PDF 파일을 올려주세요",   type=["pdf"] )
st.write("----------------")

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

import threading
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
except ImportError:
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx, add_script_run_ctx

class StreamHandler(  BaseCallbackHandler ):
    """
    GPT가 토큰을 생성할 때마다
    Streamlit 화면에 출력하는 Handler

    예:
    GPT:   안녕하세요
    생성 과정:
    안
    안녕
    안녕하세요

    처럼 실시간 출력
    """
    def __init__(  self,    container  ):
        self.container = container
        self.text = ""
        self.ctx = get_script_run_ctx()

    def on_llm_new_token(  self,  token,   **kwargs ):
        add_script_run_ctx(threading.current_thread(), self.ctx)
        # 새 토큰 누적
        self.text += token
        # 화면 갱신
        self.container.markdown(    self.text  )

if uploaded_file is not None:
    if not openai_key:
        st.warning("위 입력칸에 OpenAI API 키를 먼저 입력해주세요.")
        st.stop()

    # 캐싱(세션 상태)을 통해 페이지가 새로고침 될 때마다 DB를 다시 만드는 것을 방지합니다.
    if "db" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
        with st.spinner("최초 1회 문서를 분석하고 임베딩(DB 생성) 중입니다. 문서 크기에 따라 시간이 걸릴 수 있습니다..."):
            pages = pdf_to_document(   uploaded_file   )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            texts = text_splitter.split_documents(    pages   )

            embeddings = OpenAIEmbeddings(  api_key=openai_key   )

            db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings
            )
            
            # 세션에 저장
            st.session_state.db = db
            st.session_state.uploaded_filename = uploaded_file.name
            st.success("문서 분석이 완료되었습니다! 이제 질문해주세요.")
    else:
        # 이미 분석된 DB 재사용
        db = st.session_state.db

    retriever = db.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    st.header(   "PDF에게 질문하세요"   )
    question = st.text_input(   "질문 입력"    )

    if st.button(   "질문하기"   ):
        if question == "":
            st.warning( "질문을 입력하세요"   )
        else:
            with st.spinner(  "답변 생성중..."  ,show_time=True  ): 

                chat_box = st.empty()

                handler = StreamHandler(      chat_box       )

                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=openai_key,
                    streaming=True,
                    callbacks=[ handler  ]
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 AI 입니다.
                    Context:   {context}
                    Question:  {input}
                    답변:
                    """
                )

                document_chain = ( create_stuff_documents_chain(   llm,    prompt    )   )

                qa_chain = create_retrieval_chain(
                    retriever,
                    document_chain
                )

                qa_chain.invoke(    {    "input": question    }      )