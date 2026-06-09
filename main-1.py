# pip install -U langchain-community langchain-text-splitters pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF 로더 인스턴스 생성
loader = PyPDFLoader("novel.pdf")
# PDF 파일을 페이지 로드하고 분할하여 페이지 객체 리스트로 반환
pages = loader.load_and_split()

# Split 단계 (텍스트 조각으로 분할) 설정
# LLM이 처리하기 좋게 문서를 더 작은 단위(chunk)로 잘게 쪼갠다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, 
    # 하나의 텍스트 조각에 들어갈 최대 글자 수
  
    chunk_overlap = 20, 
    # 텍스트 조각 간의 겹치는 글자 수 (문맥 끊기는 것을 방지하기 위해 10~20% 정도 겹치게 설정함)

    length_function = len, 
    # 텍스트 길이를 계산하는 함수 (기본값은 len, 글자 수로 계산)

    is_separator_regex = False, 
    # 구분자(separator)가 정규 표현식인지 여부 (False로 설정하여 단순 문자열로 처리)
)


texts = text_splitter.split_documents(pages)
# 분할된 텍스트 조각(chunk) 객체 리스트 반환


if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])
    
    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")