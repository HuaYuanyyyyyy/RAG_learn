from itertools import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-xDasgOo6PnwafkgXunoPVfc5A0AdLuHUNkk7boNZf5A5arrD",
    base_url="https://onetoken.one/v1",
    model="gpt-4.1-nano"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}问答助手，请根据用户的问题，详细回答{role}相关的问题。"),
    ("user", "{input}")
])

chain = prompt | llm

for token in chain.stream({"role": "AI", "input": "什么是MCP"}):
    print(token.content, end="", flush=True)