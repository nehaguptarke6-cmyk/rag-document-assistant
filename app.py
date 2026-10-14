
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

load_dotenv()

st.title("📚 RAG-Based Intelligent Document Assistant")

st.write(
    "Upload PDF or TXT documents and ask questions "
    "using information retrieved from your documents."
)

st.info(
    "This assistant uses LlamaIndex for document indexing "
    "and LangChain with Groq for answer generation."
)


# ==================================================
# 2. SIDEBAR SETTINGS
# ==================================================

st.sidebar.header("Settings")

# Read the API key from .env if it exists.
saved_api_key = os.getenv("GROQ_API_KEY", "")

# Allow the user to enter a key without displaying it.
api_key = st.sidebar.text_input(
    "Groq API Key",
    value=saved_api_key,
    type="password",
    help="Your key is used for this session. Do not share it.",
)

model_name = st.sidebar.text_input(
    "Groq Model ID",
    value="openai/gpt-oss-20b",
)

st.sidebar.caption(
    "Model availability depends on your Groq account."
)


# ==================================================
# 3. CONFIGURE THE EMBEDDING MODEL
# ==================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """
    Load the Hugging Face embedding model.
    Cache it so it is not downloaded/reloaded on every rerun.
    """
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================================
# 4. CONFIGURE DOCUMENT CHUNKING
# ==================================================

splitter = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=50,
)


# ==================================================
# 5. DOCUMENT UPLOAD SECTION
# ==================================================

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

st.caption(
    "Supported formats: PDF and TXT. "
    "The document text is processed for retrieval."
)


# ==================================================
# 6. LOAD DOCUMENTS AND BUILD INDEX
# ==================================================

if st.button("Build / Rebuild Document Index", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF or TXT file.")

    else:
        try:
            with st.spinner(
                "Loading documents and building the index..."
            ):
                # Save uploaded files temporarily so the reader
                # can process them using their file paths.
                with tempfile.TemporaryDirectory() as temp_dir:

                    temp_path = Path(temp_dir)
                    input_paths = []

                    for uploaded_file in uploaded_files:
                        safe_name = Path(
                            uploaded_file.name
                        ).name

                        file_path = temp_path / safe_name

                        file_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        input_paths.append(str(file_path))

                    # Load the PDF and TXT documents.
                    documents = SimpleDirectoryReader(
                        input_files=input_paths
                    ).load_data()

                    if not documents:
                        st.error(
                            "No readable document content was found."
                        )
                        st.stop()

                    # Store the original filename in metadata
                    # so the source can be shown to the user.
                    for document in documents:
                        original_path = document.metadata.get(
                            "file_path", ""
                        )

                        document.metadata["file_name"] = (
                            Path(original_path).name
                        )

                    # Configure the embedding model.
                    Settings.embed_model = load_embedding_model()

                    # Split the documents into chunks.
                    nodes = splitter.get_nodes_from_documents(
                        documents
                    )

                    if not nodes:
                        st.error(
                            "No text chunks could be created."
                        )
                        st.stop()

                    # Build the vector index from the chunks.
                    index = VectorStoreIndex(nodes)

                    # Create a retriever that returns the top
                    # three relevant chunks for each question.
                    retriever = index.as_retriever(
                        similarity_top_k=3
                    )

                    # Keep the index and retriever available
                    # during Streamlit reruns.
                    st.session_state["index"] = index
                    st.session_state["retriever"] = retriever
                    st.session_state["document_count"] = len(
                        uploaded_files
                    )
                    st.session_state["chunk_count"] = len(nodes)

                    # Save the filenames for display.
                    st.session_state["file_names"] = [
                        Path(file.name).name
                        for file in uploaded_files
                    ]

            st.success("Document index created successfully!")

        except Exception as error:
            st.error(
                "Could not build the index. "
                "Check the uploaded files and error details."
            )
            st.exception(error)


# ==================================================
# 7. DISPLAY INDEX INFORMATION
# ==================================================

if "index" in st.session_state:

    st.header("2. Document Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Uploaded Files",
            st.session_state["document_count"],
        )

    with col2:
        st.metric(
            "Text Chunks",
            st.session_state["chunk_count"],
        )

    with st.expander("View uploaded filenames"):
        for filename in st.session_state["file_names"]:
            st.write(f"📄 {filename}")


# ==================================================
# 8. LANGCHAIN PROMPT TEMPLATE
# ==================================================

template = """
You are a helpful document question-answering assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not invent facts.
2. Use simple, clear language.
3. If the answer is not supported by the context,
   say that you could not find the information in the documents.
4. If possible, mention the relevant source filename.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# ==================================================
# 9. CREATE THE LANGCHAIN CHAIN
# ==================================================

def create_answer_chain(key, model_id):
    """
    Connect the prompt, Groq language model and output parser.
    """
    llm = ChatGroq(
        model=model_id,
        temperature=0,
        api_key=key,
    )

    # Runnable composition:
    # Prompt -> LLM -> Plain-text output parser
    chain = prompt | llm | StrOutputParser()

    return chain


# ==================================================
# 10. QUESTION-ANSWERING SECTION
# ==================================================

st.header("3. Ask Questions")

if "retriever" not in st.session_state:

    st.info(
        "Upload your documents and build the index "
        "before asking a question."
    )

else:

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is Machine Learning?",
    )

    if st.button("Get Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching your documents..."
                ):
                    # Retrieve relevant chunks.
                    retriever = st.session_state["retriever"]

                    results = retriever.retrieve(question)

                    if not results:
                        st.warning(
                            "No relevant text was retrieved."
                        )
                        st.stop()

                    # Combine retrieved text into the context.
                    context_parts = []

                    for result in results:
                        source_name = result.node.metadata.get(
                            "file_name",
                            "Unknown source",
                        )

                        context_parts.append(
                            f"Source: {source_name}\n"
                            f"Text: {result.text}"
                        )

                    context = "\n\n".join(context_parts)

                # Display the retrieved evidence.
                st.subheader("Retrieved Context")

                for number, result in enumerate(results, start=1):
                    source_name = result.node.metadata.get(
                        "file_name",
                        "Unknown source",
                    )

                    with st.expander(
                        f"Result {number} — {source_name}"
                    ):
                        st.write(result.text)

                        if result.score is not None:
                            st.caption(
                                "Retrieval similarity score: "
                                f"{result.score:.4f}"
                            )

                # Generate an answer if an API key is available.
                if not api_key.strip():

                    st.warning(
                        "No Groq API key was provided. "
                        "The retrieved context is available above, "
                        "but answer generation is disabled."
                    )

                else:
                    with st.spinner(
                        "Generating an answer with Groq..."
                    ):
                        chain = create_answer_chain(
                            api_key.strip(),
                            model_name.strip(),
                        )

                        answer = chain.invoke({
                            "context": context,
                            "question": question,
                        })

                    st.subheader("Generated Answer")
                    st.write(answer)

                    st.caption(
                        "The answer is generated from retrieved "
                        "document context. Always verify important facts."
                    )

            except Exception as error:
                st.error(
                    "The question could not be processed. "
                    "Check your API key, model ID, network, "
                    "and account limits."
                )

                # Keep troubleshooting details visible locally.
                with st.expander("Technical error details"):
                    st.exception(error)


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "RAG-Based Intelligent Document Assistant | "
    "LlamaIndex + LangChain + Hugging Face + Groq"
)

st.caption(
    "Privacy note: Uploaded documents are processed by this "
    "application. When Groq generation is enabled, retrieved "
    "context and the question are sent to the configured provider."
)
