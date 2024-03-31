#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores.faiss import FAISS
import os, sys

import nltk
nltk.download('punkt')

def knowledge_from_pdf_md(fn='./'):
    """ traverse all sub directories if any

    Args:
        fn (str, optional): _description_. Defaults to './'.
    """

    def _check_access_file(fn):
        return (os.path.exists(fn) and
            os.access(fn, os.R_OK) and
            not os.path.isdir(fn))

    def _check_access_dir(fn):
        return os.path.isdir(fn) and os.access(fn, os.R_OK | os.X_OK)


    def _process_file(fn):
        content_list = []
        if fn.endswith('.pdf'):
            print(f"Processed file: {fn}")
            content_list = PyPDFLoader(fn).load_and_split() 
        
        elif fn.endswith('.md'):
            print(f"Processed file: {fn}")
            content_list = UnstructuredMarkdownLoader(fn).load()

        return content_list
    
    contents = []
    if _check_access_file(fn):
        return _process_file()

    if not _check_access_dir(fn):
        return contents

    files = os.listdir(fn)
    for file in files:
        file = os.path.join(fn,file)
        if _check_access_file(file):
            contents += _process_file(file)
        elif _check_access_dir(file):
            contents += knowledge_from_pdf_md(file)  

    return contents

doc_path = sys.argv[1] if len(sys.argv) > 1 else "./"
doc = knowledge_from_pdf_md(doc_path)    
if not doc:
    os._exit(0) 

embeddings = GPT4AllEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
texts = text_splitter.split_documents(doc)
faiss_index = FAISS.from_documents(texts, embeddings)
faiss_index.save_local("/build/faiss")

        