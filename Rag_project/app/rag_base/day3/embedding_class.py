from openai import OpenAI
from langchain_core.embeddings import Embeddings
from typing import List

class MyEmbeddings(Embeddings):
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(
            api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
            base_url="https://onetoken.one/v1")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]




# # 测试一下
# vector = embeddings.embed_query("今天天气真好")
# print(f"向量维度：{len(vector)}")