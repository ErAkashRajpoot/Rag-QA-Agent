---
title: Agentic RAG QA Assistant
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# Retrieval-Augmented QA Agent (Agentic RAG)

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** with an **agentic reasoning layer**, combining semantic search (FAISS) and a Hugging Face transformer model (FLAN-T5).  
It retrieves relevant documents, summarizes them, and generates **step-by-step reasoning + answers**.

## ⚡ Features
- FAISS-based vector database for semantic search
- SentenceTransformer embeddings
- Hugging Face FLAN-T5 for generation
- Agentic reasoning prompt (multi-step reasoning)
- Fully local (no API keys required)

## 🛠️ Tech Stack
- **Python**, **LangChain**, **Transformers**
- **FAISS** (vector store)
- **Sentence-Transformers**
- **Hugging Face Models (Flan-T5)**

## 🚀 Usage
```bash
git clone https://github.com/ErAkashRajpoot/rag-qa-agent.git
cd rag-qa-agent
pip install -r requirements.txt
python test_agentic.py
