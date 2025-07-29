from docx import Document
import textwrap
import os

directory = r'D:\python_scripts\ollama_streamlit_web_app\docs'
pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.docx')]

documents = []

for file in pdf_files:
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    chunk_text = textwrap.wrap(text, width=500)
    
    documents.append(chunk_text)
    
print(documents)