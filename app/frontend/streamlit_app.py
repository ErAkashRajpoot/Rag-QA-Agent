import os
import streamlit as st

# Determine deployment mode
# On HF Spaces: import the RAG engine directly (single-process)
# Locally with FastAPI running: use the REST API
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "direct")  # "direct" or "api"

st.set_page_config(page_title="Agentic RAG QA", layout="wide")

st.title("🤖 Agentic RAG — QA Assistant")
st.write("Ask a question about NLP and get a context-aware, extractive answer powered by FLAN-T5 + FAISS retrieval.")


@st.cache_resource
def load_agent():
    """Load the RAG agent directly into the Streamlit process (used for HF Spaces)."""
    from app.services.rag_pipeline import AgenticRAG
    return AgenticRAG()


def query_direct(question, top_k=3):
    """Query the RAG engine directly (single-process mode)."""
    agent = load_agent()
    answer, docs = agent.query(question, k=top_k)
    sources = [
        {"source": d.metadata.get("source", "unknown"), "snippet": d.page_content[:150]}
        for d in docs
    ]
    return answer, sources


def query_api(question, top_k=3):
    """Query the RAG engine via FastAPI REST API (decoupled mode)."""
    import requests
    api_port = os.environ.get("API_PORT", "8000")
    api_url = f"http://127.0.0.1:{api_port}/api/v1/query"

    response = requests.post(
        api_url,
        json={"query": question, "top_k": top_k},
        timeout=120
    )

    if response.status_code == 200:
        data = response.json()
        return data["answer"], data["sources"]
    elif response.status_code == 400:
        st.warning("Query cannot be empty.")
        return None, None
    elif response.status_code == 503:
        st.error("The backend language model is offline or booting up.")
        return None, None
    else:
        st.error(f"Backend error: HTTP {response.status_code}")
        return None, None


# Main UI
query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            if DEPLOYMENT_MODE == "api":
                answer, sources = query_api(query)
            else:
                answer, sources = query_direct(query)

            if answer is not None:
                st.subheader("Extracted Answer")
                st.write(answer)

                st.subheader("Retrieved Knowledge Snippets")
                for i, s in enumerate(sources, 1):
                    source_name = s.get("source", "unknown") if isinstance(s, dict) else s.metadata.get("source", "unknown")
                    snippet = s.get("snippet", "...") if isinstance(s, dict) else s.page_content[:150]
                    st.markdown(f"**{i}. [{source_name}]** {snippet}...")

        except Exception as e:
            st.error(f"Error: {e}")
