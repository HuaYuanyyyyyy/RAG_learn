from embedding_class import MyEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

embedding = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one",
    model="text-embedding-v4"
)


db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_docs"
)

query = input("你想要问的问题：")

result = db.similarity_search(query,k = 3)

prompt = f"""
请根据以下内容回答问题，如果内容中没有相关信息，就说不知道。

参考内容：
{result}

问题：{query}
"""

messages = [
    SystemMessage(content="你是一个专业的知识问答助手，请根据用户的问题详细回答。"),
    HumanMessage(content=prompt),
]

llm = ChatOpenAI(
    api_key="sk-F9mZiG8A2o6lTvn50f6RJn7A12wFNzyCo9foKu3DKi2EdE5g",
    base_url="https://onetoken.one/v1",
    model="claude-sonnet-4-6"
)

for chunk in llm.stream(messages):
    print(chunk.content,end="",flush=True)

# results = db.get()
# print(results["documents"])