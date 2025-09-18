from rag_pipeline import AgenticRAG

docs = open("data/sample.txt", "r").read().split("\n")
agent = AgenticRAG(docs)

question = "What does NVIDIA design?"
answer, sources = agent.query(question)

print("\nQ:", question)
print("\nA:", answer)
print("\nSources:")
for s in sources:
    print("-", s.page_content)
