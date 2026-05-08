from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-xDasgOo6PnwafkgXunoPVfc5A0AdLuHUNkk7boNZf5A5arrD",
    base_url="https://onetoken.one/v1",
    model="gpt-4.1-nano"
)

messages = [
    SystemMessage(content="你是一个专业的AI知识问答助手，请根据用户的问题，只能回答AI相关的问题。"),
    HumanMessage(content="什么是MCP"),
]

for token in llm.stream(messages):
    print(token.content, end="", flush=True)

# response = llm.invoke(messages)
# print(response.content)