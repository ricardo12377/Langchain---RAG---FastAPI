# FastAPI - Langchain with Rag

## Overview

RAG-first is a FastAPI-based project designed to implement a Conversational Retrieval-Augmented Generation (RAG) system. It leverages LangChain's ecosystem to process and retrieve context-specific information from project-related documents, enabling users to ask questions and receive accurate, context-aware responses.

## Features

- **Conversational Retrieval-Augmented Generation**: Combines conversational AI with document retrieval to provide precise answers based on project-specific contexts.
- **Document Handling**: Dynamically loads and processes project-related documents (`frontend` or `backend`) for context-aware responses.
- **Vector Database**: Uses Chroma as a vector database to store and retrieve document embeddings.
- **Memory Management**: Maintains conversation history using a buffer memory for seamless interactions.
- **Integration with LangChain**: Utilizes LangChain components like `TextLoader`, `RecursiveCharacterTextSplitter`, and `ConversationalRetrievalChain`.
- **Ollama Models**: Employs `OllamaEmbeddings` and `ChatOllama` for embedding generation and conversational AI.

## Project Structure

- **`main.py`**: The main FastAPI application that handles user queries and processes them using the RAG pipeline.
- **`handle_project.py`**: A utility module to determine the file path of project-specific documents (`frontend` or `backend`).
- **`context/`**: Contains project-related documents (`frontend.txt`, `backend.txt`) and historical information (`history.md`).
- **`requeriments.txt`**: Lists the required Python dependencies for the project.

## How It Works

1. **User Query**: Users send a POST request to the `/ask` endpoint with a question and the project context (`frontend` or `backend`).
2. **Document Loading**: The system dynamically loads the relevant document based on the project context.
3. **Text Splitting**: Documents are split into manageable chunks using a recursive character text splitter.
4. **Embedding Generation**: The chunks are converted into embeddings using the `OllamaEmbeddings` model.
5. **Vector Database**: The embeddings are stored in a Chroma vector database for efficient retrieval.
6. **Conversational AI**: The `ChatOllama` model processes the query and retrieves relevant information from the vector database.
7. **Response**: The system returns a context-aware answer to the user.

## Installation

1. Clone the repository:

```bash
   git clone <repository-url>
   cd RAG-first
```

2. Install dependencies:

```bash
   pip install -r requeriments.txt
```

3. Run the FastAPI application:

```bash
  uvicorn main:app --reload
```
