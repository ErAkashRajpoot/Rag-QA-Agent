import os
import glob
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ── Embeddings (shared across functions) ───────────────────────────────────
def _get_embeddings():
    """Get embedding model, using GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )


# ── Document Loading ──────────────────────────────────────────────────────
def load_documents(data_path: str) -> list[Document]:
    """
    Load all .txt and .pdf files from a folder.
    Returns a list of LangChain Document objects with source metadata.
    """
    documents = []

    for txt_file in glob.glob(os.path.join(data_path, "*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(
            page_content=text,
            metadata={"source": os.path.basename(txt_file), "type": "txt"}
        ))

    for pdf_file in glob.glob(os.path.join(data_path, "*.pdf")):
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    lower_text = page_text.lower()
                    if "table of contents" in lower_text[:200] or \
                       (lower_text.count("..........") > 5) or \
                       "list of figures" in lower_text[:200]:
                        continue
                    documents.append(Document(
                        page_content=page_text,
                        metadata={
                            "source": os.path.basename(pdf_file),
                            "page": i + 1,
                            "type": "pdf"
                        }
                    ))
        except Exception as e:
            print(f"Warning: Could not load {pdf_file}: {e}")

    return documents


# ── FAISS Index Builder (with disk caching) ───────────────────────────────
def build_faiss_index(data, chunk_size=500, chunk_overlap=100, cache_dir="faiss_index"):
    """
    Build a FAISS vector store from documents.
    Caches the index to disk so subsequent runs skip the embedding step.

    Args:
        data: Folder path (str) or list of text strings.
        chunk_size: Size of text chunks for splitting.
        chunk_overlap: Overlap between chunks.
        cache_dir: Directory to cache the FAISS index.
    """
    embeddings = _get_embeddings()

    # ── Try loading cached index ──────────────────────────────────────
    if isinstance(data, str) and os.path.isdir(data) and cache_dir:
        cache_path = os.path.join(cache_dir)
        if os.path.exists(cache_path):
            print(f"Loading cached FAISS index from '{cache_path}'")
            vectorstore = FAISS.load_local(
                cache_path, embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore

    # ── Build fresh index ─────────────────────────────────────────────
    if isinstance(data, str) and os.path.isdir(data):
        documents = load_documents(data)
        if not documents:
            raise ValueError(f"No .txt or .pdf files found in '{data}'")
        print(f"Loaded {len(documents)} document(s) from '{data}'")
    elif isinstance(data, list):
        documents = [Document(page_content=d) for d in data if d.strip()]
    else:
        raise ValueError("data must be a folder path or list of text strings")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ── Save index to disk for next time ──────────────────────────────
    if cache_dir:
        vectorstore.save_local(cache_dir)
        print(f"FAISS index cached to '{cache_dir}'")

    return vectorstore
