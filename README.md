# RAG-Based Intelligent Document Assistant

## Project Overview
The RAG-Based Intelligent Document Assistant is an AI-powered application that allows users to upload PDF or TXT documents and ask questions about their content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware answers.

## Features
- Upload PDF and TXT documents.
- Process documents using LlamaIndex.
- Split documents into smaller text chunks.
- Generate embeddings using HuggingFace sentence-transformer models.
- Retrieve relevant chunks using vector search.
- Generate answers using a Groq-hosted language model.
- Use LangChain PromptTemplate and StrOutputParser.
- Display retrieved context and source information.
- Interactive interface built with Streamlit.
- Support questions based on uploaded document content.

## Technologies Used
- Python
- Streamlit
- LlamaIndex
- LangChain
- HuggingFace Embeddings
- Groq API
- Vector Search
- python-dotenv

## Project Structure

```text
rag-document-assistant/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── sample_docs/
│   └── AI_Basics.txt
├── read_document.py
├── llamaindex_load_test.py
├── chunking_embeddings.py
├── build_index.py
├── test_retrieval.py
├── langchain_prompt_test.py
└── generate_answer.py
```

## Installation

1. Clone or download this repository.
2. Open the project folder in a terminal.
3. Create and activate a Python virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Never share your API key or upload your `.env` file to GitHub.

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## How It Works

1. The user uploads PDF or TXT documents.
2. LlamaIndex loads the document content.
3. The content is split into smaller chunks.
4. HuggingFace generates embeddings for the chunks.
5. A vector index supports semantic retrieval.
6. The application retrieves relevant context for the question.
7. LangChain combines the prompt, language model, and output parser.
8. The Groq-hosted model generates an answer using the retrieved context.

## Important Note
The quality of answers depends on the uploaded documents, retrieval results, model availability, and API limits. The application is designed to answer from retrieved context, but generated answers should still be checked for accuracy.

## Learning Outcomes
- Understanding Retrieval-Augmented Generation (RAG).
- Document loading and chunking.
- Embeddings and semantic retrieval.
- Building vector indexes with LlamaIndex.
- Creating prompt pipelines with LangChain.
- Integrating a hosted language model.
- Building an interactive AI application using Streamlit.

## Author
Neha Bansal
