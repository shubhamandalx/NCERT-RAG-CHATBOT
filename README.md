# NCERT RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using NCERT PDF documents. It retrieves relevant text from a persistent vector database and uses Groq's LLM API to generate context-based answers.

## Features

- **PDF ingestion:** Load NCERT PDFs using PyPDFLoader.
- **Text chunking:** Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
- **Vector database:** Store and retrieve document embeddings using persistent ChromaDB.
- **RAG pipeline:** Retrieve relevant context before generating answers.
- **LLM integration:** Use Groq with `openai/gpt-oss-20b`.
- **Environment configuration:** Keep API keys in a `.env` file.
- **Modular codebase:** Separate modules coordinated through `main.py`.

## Tech Stack

Python · uv · LangChain · ChromaDB · Groq API · PyPDF

## Project Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd NCERT-RAG-CHATBOT
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
```

Get your API key from [Groq Console](https://console.groq.com/).

### 4. Add NCERT PDFs

Place your PDF files inside:

```text
data/pdfs/
```

### 5. Run the chatbot

```bash
uv run rag-chatbot
```

The application processes the documents, stores embeddings in ChromaDB, and uses the RAG pipeline to answer questions.

### 6. Rebuild the vector database

If you add or modify PDFs and need to recreate the index, run:

```bash
uv run rag-chatbot --rebuild
```

## How It Works

1. **Load:** Extract text from NCERT PDFs.
2. **Chunk:** Split text into smaller overlapping sections.
3. **Embed & Store:** Convert chunks into embeddings and persist them in ChromaDB.
4. **Retrieve:** Find relevant chunks for the user's question.
5. **Generate:** Send the question and retrieved context to Groq for an answer.

```text
NCERT PDFs → Chunking → Embeddings → ChromaDB
                                         ↓
User Question → Retrieval → Context + Question
                                         ↓
                              Groq LLM → Answer
```

## License

Add a license if you intend to distribute the project publicly.
