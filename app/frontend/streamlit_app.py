import streamlit as st
import requests

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="Agentic RAG QA", layout="wide")

st.title("🤖 Agentic RAG — QA Assistant")
st.write("Ask a question about NLP and get a context-aware, extractive answer powered by FLAN-T5 + FAISS retrieval.")

query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Retrieving context and compiling answer from Backend API..."):
        try:
            api_url = f"http://{settings.API_ADDRESS}:{settings.API_PORT}/api/v1/query"
            
            # Submitting the request to our newly decoupled FastAPI backend
            response = requests.post(
                api_url, 
                json={"query": query, "top_k": 3},
                timeout=60 # Prevent hanging webapp
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])
                
                # Display Results
                st.subheader("Extracted Answer")
                st.write(answer)

                st.subheader("Retrieved Knowledge Snippets")
                for i, s in enumerate(sources, 1):
                    source_name = s.get("source", "unknown metadata")
                    snippet = s.get("snippet", "...")
                    st.markdown(f"**{i}. [{source_name}]** {snippet}...")
                    
            elif response.status_code == 400:
                st.warning("Query cannot be entirely empty.")
            elif response.status_code == 503:
                st.error("The backend language model is offline or booting up.")
            else:
                st.error(f"Internal Backend Execution Error: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API Background Server. Ensure FastAPI is running.")
            logger.error("Frontend could not establish TCP connect to FastAPI local server.")
        except requests.exceptions.Timeout:
            st.error("The RAG LLM Backend took too long to formulate a response.")
            logger.warning("Streamlit Connection-Timeout hit against FastAPI.")
