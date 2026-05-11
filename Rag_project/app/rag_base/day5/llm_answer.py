from openai import OpenAI
from chromadb import chromadb
from embedding_class import MyEmbeddings

llm = OpenAI(
    api_key="sk-F9mZiG8A2o6lTvn50f6RJn7A12wFNzyCo9foKu3DKi2EdE5g",
    base_url="https://onetoken.one/v1"
)

embedding = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one",
    model="text-embedding-v4"
)

chroma_client = chromadb.PersistentClient(path="/chroma_db")
collection = chroma_client.get_collection(name="rag_test_db")

query = input("请输入你想问的问题：")

query_vector = embedding.embed_query(query)
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)

context = "\n".join(results["documents"][0])
prompt = f"""请根据以下内容回答问题，如果内容中没有相关信息，就说不知道。

参考内容：
{context}

问题：{query}
"""
for ans in llm.chat.completions.create(
    messages=[{"role":"user","content":prompt}],
    model = "claude-sonnet-4-6",
    stream= True):
    if ans.choices and ans.choices[0].delta.content:
        print(ans.choices[0].delta.content, end="", flush=True)





# print(response.choices[0].message.content)



