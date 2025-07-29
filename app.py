import streamlit as st
import ollama
import chromadb

import os
import textwrap
import fitz
from docx import Document
import io

def load_documents_from_file(uploaded_file):
    documents = []
    filename = uploaded_file.name
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        # PyMuPDF can open from bytes buffer
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        chunks = textwrap.wrap(text, width=500)
        documents.extend(chunks)

    elif ext == "docx":
        # python-docx requires a file-like object
        file_bytes = uploaded_file.read()
        file_stream = io.BytesIO(file_bytes)
        doc = Document(file_stream)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        chunks = textwrap.wrap(text, width=500)
        documents.extend(chunks)

    elif ext == "txt" or ext == "md":
        # For txt/md, decode bytes to string
        content = uploaded_file.read().decode("utf-8")
        lines = content.splitlines()
        chunks = [line.strip() for line in lines if line.strip()]
        documents.extend(chunks)

    else:
        st.warning("Unsupported file type.")
    
    return documents

def embed_and_store_documents(documents):
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")

    try:
        collection = chroma_client.get_collection(name="docs")
    except:
        collection = chroma_client.create_collection(name="docs")

    existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()

    new_count = 0
    for i, d in enumerate(documents):
        doc_id = str(i)
        if doc_id not in existing_ids:
            embedding = ollama.embed(model="phi3-custom", input=d)["embeddings"]
            collection.add(ids=[doc_id], embeddings=embedding, documents=[d])
            new_count += 1

    st.success(f"✅ Indexed {new_count} new document chunks.")
    
def save_uploaded_file(uploaded_file, save_dir="D:\\python_scripts\\ollama_streamlit_web_app\\docs"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path

st.set_page_config(page_title="Ollama RAG Chat", layout="centered")
st.title("ATOM GPT")

# --------------- 🧭 Sidebar (ChatGPT-style left panel) ---------------- #
with st.sidebar:
    st.title("Chat History") 
    st.button("New Chat")
    st.markdown("---")
    st.markdown("### Settings")
    model = st.selectbox("Choose Model", ["phi3-custom"])
    
    if "messages" in st.session_state and st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"**You:** {content[:50]}{'...' if len(content) > 50 else ''}")
                
    else:
        st.write("No chat history yet.")
    

# Load collection only once
if "collection" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    st.session_state.collection = chroma_client.get_collection(name="docs")

collection = st.session_state.collection

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial assistant message
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Ask me anything..."})
    

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# Inline file uploader (above input)
uploaded_file = st.file_uploader("Attach file", type=["txt", "pdf", "docx", "md"])

if uploaded_file:
    st.info(f"Processing {uploaded_file.name} ...")
    local_path = save_uploaded_file(uploaded_file)
    docs = load_documents_from_file(uploaded_file)
    embed_and_store_documents(docs)
    st.success(f"Uploaded and processed file: {uploaded_file.name}")

# User input
user_input = st.chat_input("Type your question...")
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.spinner("Thinking..."):
        # Get query embedding
        query_embedding = ollama.embed(model="phi3-custom", input=user_input)["embeddings"]

        # Search in Chroma collection
        results = collection.query(query_embeddings=query_embedding, n_results=1)

        context_data = results['documents'][0][0]

        # Prepare prompt with context and question
        prompt = f"Using this data: {context_data}\nAnswer the question: {user_input}"

        # Call Ollama model
        response = ollama.generate(model="phi3-custom", prompt=prompt)
        answer = response['response']

    # Save and display assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
