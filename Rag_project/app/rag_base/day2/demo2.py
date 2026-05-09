from turtle import mode
import numpy as np
from openai import OpenAI

#计算两段文字的余弦相似度

def cosine(v1,v2):
    v1,v2 = np.array(v1),np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 

client = OpenAI(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one/v1",
)


pairs = [
    ("开心", "高兴"),        # 近义词
    ("开心", "难过"),        # 反义词
    ("苹果", "香蕉"),        # 同类但不同
    ("苹果", "手机"),        # 完全不相关
    ("我喜欢吃苹果", "我喜欢吃水果"),  # 语义包含
    ("今天下雨了", "苹果手机很好用"),  # 毫无关联
]

for v1,v2 in pairs:
    cos1 = client.embeddings.create(model = "text-embedding-v4",input = v1)
    cos2 = client.embeddings.create(model = "text-embedding-v4",input = v2)
    print(f"{v1}和{v2}的近似度为{cosine(cos1.data[0].embedding,cos2.data[0].embedding)}")


# v1 = client.embeddings.create(
#     model = "text-embedding-v4",
#     input = "今天天气真好"
# )

# v2 = client.embeddings.create(
#     model = "text-embedding-v4",
#     input = "今天天气风和日丽"
# )

# v3 = client.embeddings.create(
#     model = "text-embedding-v4",
#     input = "我的游戏玩的很六"
# )

# print(cosine(v1.data[0].embedding,v2.data[0].embedding))
# print(cosine(v3.data[0].embedding,v2.data[0].embedding))

