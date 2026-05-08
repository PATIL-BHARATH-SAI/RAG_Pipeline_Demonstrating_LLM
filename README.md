# Simple RAG Pipeline Demonstrating LLM with Basics Libraries and Representing in Image Pipeline

## Overview
This repository demonstrates a simple yet powerful Retrieval-Augmented Generation (RAG) pipeline. It uses a combination of open-source embeddings, local vector storage, and a cloud-based Large Language Model (LLM) to answer user queries based on a provided PDF document. Additionally, it visualizes the underlying LangChain execution graph using both ASCII and graphical Mermaid PNG representations.

## Features
* **Document Ingestion**: Loads and processes PDF documents (e.g., *Hands-On Large Language Models*).
* **Text Chunking**: Splits large texts into manageable chunks using `RecursiveCharacterTextSplitter` (1000 characters with 100 overlap).
* **Local Embeddings & Vector Store**: Uses HuggingFace's `all-MiniLM-L6-v2` model and `FAISS` for fast, local, offline vector searches.
* **Powerful LLM Integration**: Connects to Groq's high-speed API utilizing the `llama-3.3-70b-versatile` model.
* **Source Tracking**: Transparently returns the exact source document and page number used to generate the final answer.
* **Pipeline Visualization**: Inspects the LangChain execution chain by rendering it as a text-based ASCII graph and a visual Mermaid PNG.

## Prerequisites
Ensure you have the necessary Python libraries installed to run the notebook:
```bash
pip install langchain langchain-groq langchain-community langchain-huggingface
pip install pypdf faiss-cpu sentence-transformers
pip install grandalf IPython
