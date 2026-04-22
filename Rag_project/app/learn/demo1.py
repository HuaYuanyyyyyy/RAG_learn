from openai import OpenAI
import math
client = OpenAI(
    api_key="your apikey", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#余弦相似度
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """ 计算点积，模长，把点积除以两个模长的乘积 """

    # 点积
    dot = sum(a * b for a, b in zip(v1, v2))
    # 模长
    norm1 = math.sqrt(sum(x ** 2 for x in v1))
    norm2 = math.sqrt(sum(x ** 2 for x in v2))
    return dot/(norm1 * norm2);

# 把两句话转成向量
v1 = client.embeddings.create(input="今天天气好好", model="text-embedding-v2")
v2 = client.embeddings.create(input="要是下雨就好了", model="text-embedding-v2")
v3 = client.embeddings.create(input="汽车还是大众好", model="text-embedding-v2")

# 打印向量长度
print(len(v1.data[0].embedding))  # 1536维

#计算余弦相似度
print(cosine_similarity(v1.data[0].embedding,v2.data[0].embedding))
print(cosine_similarity(v3.data[0].embedding,v2.data[0].embedding))
