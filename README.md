# RAG
 
# PDF Embedding and Querying with FAISS & LLM
This repository provides an end-to-end pipeline to:

Parse PDFs

Chunk and embed content with sentence-transformers

Store and search using FAISS vector store

Re-rank results with a cross-encoder

Query contextually using an LLM (such as Gemini via OpenAI-compatible API)

# Files
ext_faiss.py
Indexes all PDFs in a data directory into a FAISS vector database.

query.py
Loads the index, retrieves relevant chunks for a query, reranks them, and queries an LLM for an answer.

# Requirements
Python 3.8+

langchain

sentence-transformers

PyPDF2

faiss-cpu

python-dotenv

An OpenAI-compatible LLM endpoint (e.g., Google's Gemini, local LLMs, etc.)

PDF files in the data/ directory