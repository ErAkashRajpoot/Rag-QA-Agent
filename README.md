# Retrieval-Augmented QA Agent (Agentic RAG)

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** with an **agentic reasoning layer**, combining semantic search (FAISS) and a Hugging Face transformer model (FLAN-T5).  
It retrieves relevant documents from NLP knowledge base, and generates **factual, extractive answers**.

## ⚡ Features
- FAISS-based vector database for semantic search
- SentenceTransformer embeddings (`all-mpnet-base-v2`)
- Hugging Face FLAN-T5 for generation
- Few-shot extractive QA prompt
- 100 curated NLP topic documents covering foundations to advanced topics
- FAISS index caching for fast restarts
- Fully local (no API keys required)

## 🛠️ Tech Stack
- **Python**, **LangChain**, **Transformers**
- **FAISS** (vector store)
- **Sentence-Transformers**
- **Hugging Face Models (FLAN-T5-large)**


## 📂 Data
The `data/` folder contains `NLP DOC.txt` — 100 structured documents covering:
- NLP Foundations, Tokenization, Embeddings
- Transformers, BERT, GPT, Attention Mechanisms
- RAG Pipelines, Retrieval, Vector Search
- Fine-tuning, LoRA, Transfer Learning
- Ethics, Bias, Privacy in NLP

## 🚀 Usage
```bash
git clone https://github.com/ErAkashRajpoot/rag-qa-agent.git
cd rag-qa-agent
pip install langchain langchain-community langchain-huggingface transformers faiss-cpu sentence-transformers
python test_agentic.py
```


## 🧪 Example Questions
```
What is tokenization?
What are the main gates in LSTM?
What is Retrieval-Augmented Generation?
What is the attention mechanism?
What are common fine-tuning strategies?
```
