from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

embeddings = GPT4AllEmbeddings(model="/models/all-MiniLM-L6-v2-f16.gguf")
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


question = 'is routine cleaning covered in dental benefits?'
matched_docs = faiss_index.similarity_search(question, 4)
context = ""
for doc in matched_docs:
    context = context + doc.page_content + " \n\n " 

llm = GPT4All(model="/models/mistral-7b-openorca.gguf2.Q4_0.gguf", max_tokens=2000)

template = """
Please use the following context to answer questions.
Context: {context}
 - -
Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.invoke(question))