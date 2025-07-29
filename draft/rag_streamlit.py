import streamlit as st
import ollama
import chromadb
import textwrap
import fitz  # PyMuPDF
import os
from docx import Document

UPLOAD_DIR = r'D:\python_scripts\ollama_streamlit_web_app\docs'
client = chromadb.Client()
documents = []

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_documents():
    documents.clear()
    
    # Load PDF files
    pdf_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
    for file in pdf_files:
        doc = fitz.open(file)
        text = "\n".join([page.get_text() for page in doc])
        chunk_text = textwrap.wrap(text, width=500)
        documents.extend(chunk_text)
    
    # Load DOCX files
    docx_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith('.docx')]
    for file in docx_files:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        chunk_text = textwrap.wrap(text, width=500)
        documents.extend(chunk_text)

    # Load TXT files
    txt_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith('.txt')]
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            chunk_text = textwrap.wrap(text, width=500)
            documents.extend(chunk_text)

def embed_documents():
    try:
        collection = client.get_collection(name="docs")
    except:
        collection = client.create_collection(name="docs")

    for i, d in enumerate(documents):
        response = ollama.embed(model="phi3-custom", input=d)
        embeddings = response["embeddings"]
        
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d],
            metadatas=[{"source": file}]
        )
    return True

# Streamlit UI
st.title("📄 Document Q&A with Ollama + ChromaDB")

# Upload section
uploaded_files = st.file_uploader("Upload documents (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("Files uploaded successfully.")

    load_documents()
    if embed_documents():
        st.success("Documents embedded into vector database.")

# Ask a question
question = st.text_input("Ask a question based on uploaded documents:")

if question:
    try:
        collection = client.get_collection(name="docs")
    except:
        st.error("No documents embedded yet.")
        st.stop()

    query_embedding = ollama.embed(model="phi3-custom", input=question)["embeddings"]

    results = collection.query(query_embeddings=query_embedding, n_results=3)
    if results['documents']:
        source_data = results['documents'][0][0]
        output = ollama.generate(
            model="phi3-custom",
            prompt=f"Using this data: {source_data}. Respond to this prompt: {question}"
        )
        st.subheader("💬 Answer")
        st.write(output["response"])
        st.markdown("##### 📄 Source Document Snippet")
        st.code(source_data)
    else:
        st.warning("No relevant documents found.")
