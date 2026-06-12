import os
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드 (.env 파일 내 OPENAI_API_KEY 확인)
load_dotenv()

st.set_page_config(page_title="부동산 데이터 자연어 검색", page_icon="🏢", layout="wide")

st.title("🏢 부동산 데이터 자연어 검색 (CSV 기반 RAG)")
st.markdown("공공데이터(CSV)를 업로드하고 자연어로 질문해보세요. AI가 문맥을 파악하여 알맞은 데이터를 찾아 답변합니다.")
st.write("---")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "df" not in st.session_state:
    st.session_state.df = None

# 사이드바: 설정 및 파일 업로드
with st.sidebar:
    st.header("⚙️ 설정 및 파일 업로드")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    uploaded_file = st.file_uploader("CSV 파일 업로드 (cp949/euc-kr 인코딩 권장)", type=["csv"])
    
    st.divider()
    if st.button("대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 파일 업로드 시 벡터 DB 구축
if uploaded_file and openai_api_key:
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("데이터를 분석하고 벡터 스토어를 구축 중입니다... 잠시만 기다려주세요."):
            # CSVLoader를 사용하기 위해 업로드된 파일을 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 1. 파일 인코딩 및 상단 불필요한 메타데이터(안내문) 스킵 처리
                with open(tmp_file_path, 'r', encoding='cp949') as f:
                    lines = f.readlines()
                
                # 실제 데이터(헤더)가 시작하는 위치 찾기 (보통 쉼표가 여러 개 있는 행)
                header_idx = 0
                for i, line in enumerate(lines):
                    if line.count(',') > 5: # 열(컬럼)이 5개 초과인 행을 헤더로 간주
                        header_idx = i
                        break
                
                # 실제 데이터만 추출하여 새로운 임시 파일(utf-8)로 저장
                clean_tmp_file_path = tmp_file_path + "_clean.csv"
                with open(clean_tmp_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[header_idx:])
                
                # 사용자 요약 정보 제공용으로 Pandas로 읽어 세션에 저장 (정제된 파일 사용)
                st.session_state.df = pd.read_csv(clean_tmp_file_path, encoding="utf-8")
                
                # LangChain CSVLoader 로드 (정제된 파일 사용)
                loader = CSVLoader(file_path=clean_tmp_file_path, encoding="utf-8")
                documents = loader.load()
                
                # 임베딩 및 Chroma DB 생성
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
                db = Chroma.from_documents(documents, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 4})
                
                # LLM 및 프롬프트 설정
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
                
                prompt = ChatPromptTemplate.from_template(
                    """당신은 부동산 데이터를 분석하고 친절하게 안내하는 AI 어시스턴트입니다.
                    아래 제공된 Context만을 사용하여 사용자의 질문에 답변하세요.
                    문맥에 없는 내용은 절대 지어내지 마세요. 답변은 가독성이 좋게 작성해 주세요.
                    
                    Context: {context}
                    
                    Question: {input}
                    
                    답변:"""
                )
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.qa_chain = create_retrieval_chain(retriever, document_chain)
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = [] # 새 파일 로드 시 대화 초기화
                
                st.sidebar.success("✅ 데이터 학습 완료!")
            except Exception as e:
                st.sidebar.error(f"❌ 데이터 처리 중 오류 발생: {e}")
            finally:
                # 처리 완료 후 임시 파일 삭제
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                if 'clean_tmp_file_path' in locals() and os.path.exists(clean_tmp_file_path):
                    os.remove(clean_tmp_file_path)

# 챗봇 UI 메인 영역
if st.session_state.qa_chain:
    # --- 데이터 요약 정보 제공 UI ---
    if st.session_state.df is not None:
        with st.expander("📊 업로드된 데이터 구조 한눈에 보기 (질문 힌트)", expanded=False):
            st.write(f"총 **{len(st.session_state.df)}**개의 데이터(Row)가 학습되었습니다.")
            st.markdown(f"**🔹 질문 가능한 항목(컬럼):** `{'`, `'.join(st.session_state.df.columns.tolist())}`")
            st.markdown("**🔹 데이터 미리보기 (상위 5개)**")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
        st.info("""
        **💡 추천 질문 샘플**
        - **질문 1:** 2024년 3월 서울 종로구 명륜2가에 있는 아남1 아파트의 건축년도와 도로명 주소 정보는 무엇인가요?
        - **질문 2:** 명륜2가 아남1 아파트의 거래 내역 중 거래금액이 125,000만 원 이상이거나 3월 25일 이후에 계약된 건의 층수는 각각 몇 층인가요?
        - **질문 3:** 2024년 3월에 계약된 명륜2가 아남1 아파트(전용면적 84.9㎡)의 최고 거래금액과 최저 거래금액은 각각 얼마이며, 두 거래의 계약일은 며칠인지 비교하여 요약해 줄 수 있나요?
        """)
    st.divider()

    # 1. 채팅 히스토리 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # AI 답변인 경우 참조된 데이터(Context)를 토글 형태로 표시
            if msg.get("context"):
                with st.expander("🔍 AI가 참조한 데이터 원본 확인"):
                    for i, doc in enumerate(msg["context"], 1):
                        st.markdown(f"**[데이터 {i}]**\n```text\n{doc.page_content}\n```")

    # 2. 채팅 입력창
    if query := st.chat_input("예: 2024년 3월 아남1 아파트의 건축년도와 도로명 주소는?"):
        # 사용자 메시지 화면 표시 및 세션 저장
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # AI 답변 화면 표시 및 세션 저장
        with st.chat_message("assistant"):
            with st.spinner("문맥을 파악하여 데이터를 검색하고 있습니다..."):
                response = st.session_state.qa_chain.invoke({"input": query})
                answer = response["answer"]
                contexts = response["context"]
                
                st.markdown(answer)
                with st.expander("🔍 AI가 참조한 데이터 원본 확인"):
                    for i, doc in enumerate(contexts, 1):
                        st.markdown(f"**[데이터 {i}]**\n```text\n{doc.page_content}\n```")
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": contexts
                })
else:
    # 파일 로드가 되지 않은 상태에서의 초기 안내
    if not openai_api_key:
        st.info("👈 왼쪽 사이드바에 OpenAI API Key를 입력해주세요.")
    elif not uploaded_file:
        st.info("👈 왼쪽 사이드바에서 분석할 CSV 파일을 업로드해주세요.")