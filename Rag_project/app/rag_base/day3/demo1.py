from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from embedding_class import MyEmbeddings




embeddings = MyEmbeddings(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one/v1",
    model="text-embedding-v4"
)

#文档
docs = [
    "氮氧化物的排放限值为100mg/m³，超过此标准视为不合规",
    "二氧化硫排放浓度不得超过50mg/m³",
    "PM2.5年均浓度限值为35微克每立方米",
    "工业废水中化学需氧量不得超过100mg/L",
    "噪声排放标准：昼间不超过65分贝，夜间不超过55分贝",
    "今天中午吃了一碗面条",
    "Python是目前最流行的编程语言之一",
]

db = Chroma.from_texts(
    texts = docs,
    embedding=embeddings,
    persist_directory= "./chroma_db"
)

print(db._collection.count())


