import chromadb
client = chromadb.PersistentClient(path="/chroma_db")
collection = client.get_collection(name = "rag_test_db")
print(collection.get()["documents"][0])