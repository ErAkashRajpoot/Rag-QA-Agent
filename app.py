import os
from huggingface_hub import login
import streamlit as st
from rag_pipeline import AgenticRAG

# --- Hugging Face Authentication ---
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

st.set_page_config(page_title="Agentic RAG QA", layout="wide")

@st.cache_resource
def load_agent():
    docs = open("data/sample.txt", "r", encoding="utf-8").read().split("\n")
    return AgenticRAG(docs)

agent = load_agent()

st.title("🤖 Agentic RAG — QA Assistant")
st.write("Ask a question and get a context-aware, step-by-step answer (agentic reasoning).")

query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Thinking… retrieving context and generating answer"):
        answer, sources = agent.query(query)

    st.subheader("Answer")
    try:
        st.write(answer if isinstance(answer, str) else str(answer))
    except Exception:
        st.write(str(answer))

    st.subheader("Source snippets")
    for i, s in enumerate(sources, 1):
        st.markdown(f"**{i}.** {s.page_content}")
