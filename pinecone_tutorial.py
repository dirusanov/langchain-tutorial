import os

import pinecone
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv(find_dotenv())


PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']


def doc_preprocessing_pdf():
    loader = DirectoryLoader(
        './data',
        glob='**/*pdf',
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


def doc_preprocessing_web():
    loader = WebBaseLoader("https://d-rusanov.ru/about")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing_pdf()
    return Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='langchain-demo-indexes'
    )


llm = ChatOpenAI(model_name='gpt-4-1106-preview')
doc_db = embedding_db()


def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result


def main():
    st.title("Question and Answering App powered by LLM and Pinecone")
    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)


if __name__ == "__main__":
    main()
