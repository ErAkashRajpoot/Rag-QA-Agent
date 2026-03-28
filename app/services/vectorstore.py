import os
import glob
import hashlib
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)


def _compute_directory_hash(directory: str) -> str:
    """Computes a secure hash based on the file sizes and modification times of all files in a directory."""
    hasher = hashlib.sha256()
    if not os.path.exists(directory):
        return hasher.hexdigest()

    for root, _, files in os.walk(directory):
        for names in sorted(files):
            filepath = os.path.join(root, names)
            try:
                stat = os.stat(filepath)
                # Combine filepath string, size, and modified time for a deterministic state footprint
                state_str = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
                hasher.update(state_str.encode('utf-8'))
            except FileNotFoundError:
                continue
    return hasher.hexdigest()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate embedding model via settings, utilizing GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Loading embeddings model '{settings.EMBEDDING_MODEL}' on device '{device}'")
    
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )


def load_documents(data_path: str) -> list[Document]:
    """
    Load all .txt and .pdf files from a folder robustly.
    Skips unreadable files with a logged warning.
    """
    documents = []

    # Processing text files
    for txt_file in glob.glob(os.path.join(data_path, "*.txt")):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(txt_file), "type": "txt"}
            ))
        except Exception as e:
            logger.error(f"Failed to read TXT file {txt_file}: {e}")

    # Processing PDF files
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf dependency missing. Skipping PDF ingestion.")
        return documents

    for pdf_file in glob.glob(os.path.join(data_path, "*.pdf")):
        try:
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    lower_text = page_text.lower()
                    
                    # Heuristic Skip for Table of Contents and empty dotted structures
                    if "table of contents" in lower_text[:200] or (lower_text.count("..........") > 5) or "list of figures" in lower_text[:200]:
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
            logger.error(f"Failed to load PDF {pdf_file}: {e}")

    return documents


def get_faiss_index(
    data_path: str = settings.DATA_PATH, 
    chunk_size: int = settings.CHUNK_SIZE, 
    chunk_overlap: int = settings.CHUNK_OVERLAP, 
    cache_dir: str = settings.FAISS_CACHE_DIR
) -> FAISS:
    """
    Build or Load a FAISS vector store. 
    Implements dynamic data hashing to rebuild automatically if local files change.
    """
    embeddings = _get_embeddings()
    current_hash = _compute_directory_hash(data_path)
    hash_file_path = os.path.join(cache_dir, "dir_hash.txt")

    # 1. Verification of state
    if os.path.exists(cache_dir) and os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as f:
            cached_hash = f.read().strip()
            
        if cached_hash == current_hash:
            logger.info(f"Data directory '{data_path}' has not changed. Loading cached FAISS index.")
            try:
                return FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.error(f"Failed to load FAISS index from cache: {e}. Rebuilding...")
        else:
            logger.info("Data directory contents changed. Rebuilding FAISS index.")
    elif os.path.exists(cache_dir):
        logger.info("Cache index exists without a valid hash tag. Rebuilding for safety.")

    # 2. Rebuilding index if verification failed
    if not os.path.isdir(data_path):
        logger.warning(f"Data directory '{data_path}' does not exist. Creating it.")
        os.makedirs(data_path, exist_ok=True)
        return FAISS.from_documents([], embeddings) # Will raise errors upstream if empty, handled in API

    documents = load_documents(data_path)
    if not documents:
        raise ValueError(f"No .txt or .pdf files found in '{data_path}'. Cannot build valid index.")
        
    logger.info(f"Loaded {len(documents)} document(s) from '{data_path}'")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split raw documents into {len(chunks)} contextual chunks.")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 3. Save to disk 
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(cache_dir)
    
    with open(hash_file_path, "w") as f:
        f.write(current_hash)
        
    logger.info(f"FAISS index (Hash: {current_hash[:8]}) cached sequentially to '{cache_dir}'")

    return vectorstore
