import streamlit as st
import ollama
import chromadb

st.set_page_config(page_title="Ollama RAG Chat", layout="centered")
st.title("🧠 RAG Chatbot (Ollama + ChromaDB)")

# Load collection only once
if "collection" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    st.session_state.collection = chroma_client.get_collection(name="docs")

collection = st.session_state.collection

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial assistant message
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Ask me anything about your documents."})

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your question...")
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

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
