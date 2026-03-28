from app.services.rag_pipeline import AgenticRAG

def test_pipeline_queries():
    # Load document pipeline implicitly
    agent = AgenticRAG()

    # Test questions - generic
    questions = [
        "What is NLP?",
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

if __name__ == "__main__":
    test_pipeline_queries()
