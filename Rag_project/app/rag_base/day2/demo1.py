from openai import OpenAI

#跑第一个embedding
client = OpenAI(
    api_key="sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH",
    base_url="https://onetoken.one/v1",
)

embeddings = client.embeddings.create(
    model="text-embedding-v4",
    input="今天天气真好"
)

vector = embeddings.data[0].embedding
print(len(vector))
print("前五个元素为：", vector[:5])