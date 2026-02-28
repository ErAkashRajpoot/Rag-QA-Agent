from vectorstore import build_faiss_index

# Build FAISS index from the data folder (loads all PDFs + TXT files)
vectorstore = build_faiss_index("data")

# Try a query
query = "What is NLP?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Found {len(results)} results:\n")
for i, r in enumerate(results, 1):
    source = r.metadata.get("source", "unknown")
    page = r.metadata.get("page", "")
    page_str = f" (p.{page})" if page else ""
    print(f"  {i}. [{source}{page_str}]: {r.page_content[:150]}...")
