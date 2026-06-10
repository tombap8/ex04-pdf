# import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
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


# 사용자가 '종료'를 입력할 때까지 계속 질문을 받음
while True:
    # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
    question = input("질문을 입력하세요 (종료하려면 '종료' 입력): ")
    
    # 사용자가 '종료'를 입력하면 프로그램 종료
    if question == "종료":
        print("프로그램을 종료합니다.")
        break
    
    response = rag_chain.invoke({"input": question})
    
    # 검색된 참조 문서 개수와 최종 답변 출력
    print(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
    print(f"답변: {response['answer']}")
    print()
