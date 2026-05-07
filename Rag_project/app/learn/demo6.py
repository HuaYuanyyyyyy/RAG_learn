import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# ✅ 正确方式：获取集合对象后使用 .name 属性
collections = client.list_collections()
print("所有集合:", [col.name for col in collections])  # 提取名称

# 遍历集合
for col in collections:  # col 已经是 Collection 对象
    print(f"\n{'='*60}")
    print(f"集合: {col.name}")  # 使用 .name 获取名称
    print(f"文档数: {col.count()}")
    print('='*60)
    
    # 获取文档
    data = col.get(limit=20)  # 直接使用 col 对象
    
    for i, (doc_id, doc) in enumerate(zip(data['ids'], data['documents']), 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"内容: {doc[:150]}...")
        print("-"*40)