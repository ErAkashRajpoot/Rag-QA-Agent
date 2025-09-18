import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS   
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings       

def build_faiss_index(docs: list[str]):
    """
    Takes a list of raw text docs and returns a FAISS vector store retriever.
    """
    # Convert text docs to LangChain Document objects
    documents = [Document(page_content=d) for d in docs]

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create embeddings using OpenAI (requires OPENAI_API_KEY in env)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
