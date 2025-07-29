import textwrap
import fitz
import os

directory = r'D:\python_scripts\ollama_streamlit_web_app\docs'
pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

documents = []

for file in pdf_files:
    doc = fitz.open(file)
    text = "\n".join([page.get_text() for page in doc])

    chunk_text = textwrap.wrap(text, width=500)
    
    documents.append(chunk_text)