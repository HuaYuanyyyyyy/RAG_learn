# test_chromadb.py
import chromadb

# 创建客户端
client = chromadb.Client()

# 创建集合
collection = client.create_collection("test")

# 添加数据
collection.add(
    documents=["这是测试文档"],
    ids=["test1"]
)

# 查询
results = collection.query(
    query_texts=["测试"],
    n_results=1
)

print("ChromaDB 安装成功！")
print(f"版本: {chromadb.__version__}")
print(f"查询结果: {results}")