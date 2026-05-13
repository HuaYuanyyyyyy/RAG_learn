# 加载文本->切块->embedding->存Chroma->检索->喂给LLM->输出答案
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from embedding_class import MyEmbeddings
from langchain_chroma import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data.pdf")
loader = PyPDFLoader(file_path)
pdf_data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50
)

chunks = splitter.split_documents(pdf_data)

embedding = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one",
    model="text-embedding-v4"
)

chroma = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db",
    collection_name="pdf_docs"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_docs"
)
print(db)