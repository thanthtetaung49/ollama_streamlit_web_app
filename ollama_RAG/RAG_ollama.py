import ollama
import chromadb
from chromadb.config import Settings
import textwrap
import fitz  # PyMuPDF
import os
from docx import Document

# Specify persistent directory for chromadb
chroma_client = chromadb.PersistentClient(
    path="./chroma_storage"  # or an absolute path
)

directory = r'D:\python_scripts\ollama_streamlit_web_app\docs'
documents = []

def pdf_loader():
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    for file in pdf_files:
        doc = fitz.open(file)
        text = "\n".join([page.get_text() for page in doc])
        chunk_text = textwrap.wrap(text, width=500)
        documents.extend(chunk_text)

def docx_loader():
    docx_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.docx')]
    for file in docx_files:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        chunk_text = textwrap.wrap(text, width=500)
        documents.extend(chunk_text)

def txt_loader():
    txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            chunk_text = textwrap.wrap(text, width=500)
            documents.extend(chunk_text)

# Load all document types
pdf_loader()
docx_loader()
txt_loader()

# Create or load existing persistent collection
try:
    collection = chroma_client.get_collection(name="docs")
except:
    collection = chroma_client.create_collection(name="docs")

# Generate and store embeddings only if not already stored
existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
for i, d in enumerate(documents):
    doc_id = str(i)
    if doc_id not in existing_ids:
        response = ollama.embed(model="phi3-custom", input=d)
        embedding = response["embeddings"]
        collection.add(ids=[doc_id], embeddings=embedding, documents=[d])

# Persist the collection to disk
# chroma_client.persist()
print("Embedding and storage successful.")

# === Handle user query ===
user_query = "What are the input sources for the AEP system?"
query_embedding = ollama.embed(model="phi3-custom", input=user_query)["embeddings"]

results = collection.query(query_embeddings=query_embedding, n_results=1)
data = results['documents'][0][0]

response = ollama.generate(
    model="phi3-custom",
    prompt=f"Using this data: {data}. Respond to this prompt: {user_query}"
)

print(response['response'])
