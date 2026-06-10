from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
loader = PyPDFLoader("novel.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap = 20,         # 청크 간에 겹칠 글자 수 (문맥 단절 방지)
    length_function = len,      # 길이를 측정할 함수 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호(separator)를 정규표현식으로 해석할지 여부
)

texts = text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings_model)