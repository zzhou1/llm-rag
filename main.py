from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os
from flask import Flask, render_template, request



def set_up_embeddings():
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
    return FAISS.from_documents(texts, embeddings)

faiss_index = set_up_embeddings()
app = Flask(__name__)

@app.route('/render', methods=['POST'])
def invoke():
    question = request.form["input"]
    matched_docs = faiss_index.similarity_search(question, 4)
    context = ""
    sources = set()
    for doc in matched_docs:
        context = context + doc.page_content + " \n\n " 
        sources.add(doc.metadata["source"])
    llm = GPT4All(model="/models/mistral-7b-openorca.gguf2.Q4_0.gguf", max_tokens=2000)

    template = """
    Use the following context to answer questions, but don't be limited by it.
    Context: {context}
    - -
    Question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.invoke(question)
    return render_template('answer.html', question=question, answer=answer['text'], context=sources)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)