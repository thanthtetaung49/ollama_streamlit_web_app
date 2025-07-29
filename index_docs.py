import os
import textwrap
import fitz
from docx import Document
import ollama
import chromadb


def load_documents(directory="docs"):
    documents = []

    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        ext = file.lower().split(".")[-1]

        if ext == "pdf":
            doc = fitz.open(full_path)
            text = "\n".join([page.get_text() for page in doc])
            chunks = textwrap.wrap(text, width=500)
            documents.extend(chunks)

        elif ext == "docx":
            doc = Document(full_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            chunks = textwrap.wrap(text, width=500)
            documents.extend(chunks)

        elif ext == "txt":
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Keep short lines like "[charged_msisdns] - a_party number"
                chunks = [line.strip() for line in lines if line.strip()]
                documents.extend(chunks)

        else:
            continue
        
    
    print(documents)
    print(f"Loaded and chunked {len(documents)} document parts.")
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

    print(f"✅ Indexed {new_count} new document chunks.")


if __name__ == "__main__":
    docs = load_documents(directory="docs")
    embed_and_store_documents(docs)
