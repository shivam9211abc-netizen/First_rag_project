import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Page configuration
st.set_page_config(page_title="PDF RAG Search Engine", page_icon="📄", layout="centered")

st.title("📄 PDF Semantic Search (RAG)")
st.write("Upload a PDF document, ask a question, and retrieve the most relevant sections using your local Chroma vector store and HuggingFace embeddings.")

# Sidebar for configuration
st.sidebar.header("Configuration")
persist_directory = st.sidebar.text_input("Chroma DB Directory", value="./chroma_db")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200, step=50)
top_k = st.sidebar.slider("Top K Results", 1, 10, 5)

# Main interface: File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Query input
query = st.text_input("Enter your query:", value="What is the attention mechanism?")

if st.button("Search PDF", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a PDF file first.")
    elif not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing PDF and searching vector database..."):
            try:
                # Save uploaded file temporarily so PyPDFLoader can read it from path
                temp_pdf_path = os.path.join(".", uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Your core functional logic
                loader = PyPDFLoader(temp_pdf_path)
                document = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = splitter.split_documents(document)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )

                vector_store = Chroma.from_documents(
                    embedding=embeddings,
                    documents=chunks,
                    persist_directory=persist_directory
                )

                results = vector_store.similarity_search(query, k=top_k)

                # Clean up temporary PDF file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

                # Display Results
                st.success(f"Successfully retrieved top {len(results)} results for: '{query}'")
                st.markdown("---")

                for i, docs in enumerate(results):
                    page_num = docs.metadata.get("page", "N/A")
                    with st.expander(f"Result {i + 1} (Page {page_num})", expanded=(i == 0)):
                        st.markdown(f"**Content Snippet:**")
                        st.text(docs.page_content[:400] + "...")

            except Exception as e:
                st.error(f"An error occurred: {e}")