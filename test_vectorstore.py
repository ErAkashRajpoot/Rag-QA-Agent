from vectorstore import build_faiss_index

# Load sample docs
docs = open("data/sample.txt", "r").read().split("\n")

# Build FAISS index
vectorstore = build_faiss_index(docs)

# Try a query
query = "What does NVIDIA design?"
retriever = vectorstore.as_retriever()
results = retriever.get_relevant_documents(query)

print("\nQuery:", query)
print("Top result:", results[0].page_content)
