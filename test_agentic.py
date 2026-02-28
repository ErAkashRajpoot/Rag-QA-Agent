from rag_pipeline import AgenticRAG

# Load all documents (PDFs + TXT) from the data folder
agent = AgenticRAG("data")

# Test questions - NLP topics from NLP DOC.txt
questions = [
    "What is tokenization?",
    "What are the main gates in LSTM?",
    "What is Retrieval-Augmented Generation?",
    "What is the attention mechanism?",
    "What are common fine-tuning strategies?",
]

for question in questions:
    answer, sources = agent.query(question)

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"\nA: {answer}")
    print(f"\nSources ({len(sources)} retrieved):")
    for s in sources:
        source_info = s.metadata.get("source", "unknown")
        page = s.metadata.get("page", "")
        page_str = f" (p.{page})" if page else ""
        print(f"  - [{source_info}{page_str}]: {s.page_content[:100]}...")
    print(f"{'='*60}")
