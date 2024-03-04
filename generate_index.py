from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

embeddings = GPT4AllEmbeddings()

pdf_dir = 'pdfs'
pdf_files = os.listdir(pdf_dir)

print("reading input files")
documents = []
for pdf_file in pdf_files:
    if pdf_file.endswith('.pdf'):
        print(f"    {pdf_file}")
        pdf_path = os.path.join(pdf_dir, pdf_file)  
        documents += PyPDFLoader(pdf_path).load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
texts = text_splitter.split_documents(documents)
faiss_index = FAISS.from_documents(texts, embeddings)
faiss_index.save_local("/build/faiss")