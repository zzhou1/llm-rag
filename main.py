from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from gpt4all import Embed4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

embeddings = GPT4AllEmbeddings()

pdf_dir = 'pdfs'
pdf_files = os.listdir(pdf_dir)

all_texts = []
documents = []
for pdf_file in pdf_files:
    if pdf_file.endswith('.pdf'):
        print(f"-----------processing {pdf_file}")
        pdf_path = os.path.join(pdf_dir, pdf_file)  
        documents += PyPDFLoader(pdf_path).load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
texts = text_splitter.split_documents(documents)

faiss_index = FAISS.from_documents(texts, embeddings)

# def get_embeddings():
#     embedder = Embed4All()
#     pdf_dir = 'pdfs'
#     pdf_files = os.listdir(pdf_dir)
#     embeddings = []
#     for pdf_file in pdf_files:
#         if pdf_file.endswith('.pdf'):
#             print(f"-----------processing {pdf_file}")
#             pdf_path = os.path.join(pdf_dir, pdf_file)
            
#             with open(pdf_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 pdf_text = ""
#                 for page in pdf_reader.pages:
#                     pdf_text += page.extract_text() + "\n"
#                 output = embedder.embed(pdf_text)
#                 embeddings.append(output)
#     return embeddings

# embeddings = get_embeddings()