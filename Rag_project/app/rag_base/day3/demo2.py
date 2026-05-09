# 下次启动不用重新存，直接加载
from langchain_chroma import Chroma

from embedding_class import MyEmbeddings

embeddings = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one/v1",
    model="text-embedding-v4"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 查询
query = "java也是最好的语言之一"
results = db.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"第{i+1}条：{doc.page_content}")