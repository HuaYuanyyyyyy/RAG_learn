import os
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_class import MyEmbeddings
import chromadb


llm = OpenAI(
    api_key="sk-F9mZiG8A2o6lTvn50f6RJn7A12wFNzyCo9foKu3DKi2EdE5g",
    base_url="https://onetoken.one/v1/messages",
)

embedding = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one",
    model="text-embedding-v4"
)


#读取文件 txt为案例
text = ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "test.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

#分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

#转换向量
vectors = embedding.embed_documents(chunks)

#存入数据库

#创建客户端
chroma_client = chromadb.PersistentClient(path="/chroma_db")
#创建集合
collection = chroma_client.get_or_create_collection(name = "rag_test_db")
#存入
collection.add(
    ids = [str(i) for i in range(len(chunks))],
    embeddings= vectors,
    documents=chunks
)

print(f"存入成功，共{collection.count()}条")

