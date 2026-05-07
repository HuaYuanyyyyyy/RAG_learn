import chromadb

# 连接数据库
client = chromadb.PersistentClient(path="./chroma_db")

# 列出所有集合名称
collections = client.list_collections()
print(f"共有 {len(collections)} 个集合:")
for col in collections:
    print(f"  - {col}")