# pip install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

file_path = "novel.pdf"
loader = PyPDFLoader(file_path)

pages = loader.load_and_split()

# print(pages)
# print(pages[0])
print('전체 페이지 수:', len(pages))
print('첫 페이지 내용:', pages[0].page_content)
print('첫 페이지 메타데이터:', pages[0].metadata)
print('마지막 페이지 내용:', pages[-1].page_content)
print('마지막 페이지 메타데이터:', pages[-1].metadata)