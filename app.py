import streamlit as st
from rag_pipeline import AgenticRAG

st.set_page_config(page_title="Agentic RAG QA", layout="wide")


@st.cache_resource
def load_agent():
    return AgenticRAG("data")


agent = load_agent()

st.title("🤖 Agentic RAG — QA Assistant")
st.write("Ask a question about NLP and get a context-aware, extractive answer powered by FLAN-T5 + FAISS retrieval.")

query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Retrieving context and generating answer..."):
        answer, sources = agent.query(query)

    st.subheader("Answer")
    st.write(answer if isinstance(answer, str) else str(answer))

    st.subheader("Source snippets")
    for i, s in enumerate(sources, 1):
        source = s.metadata.get("source", "unknown")
        st.markdown(f"**{i}. [{source}]** {s.page_content[:200]}...")
