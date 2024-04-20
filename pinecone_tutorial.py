import os

from pinecone import Pinecone as PineconeClient
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain_community.vectorstores import Pinecone
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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
    PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
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
    st.title("LLM с дополнительными данными!")
    text_input = st.text_input("Задай свой вопрос...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)


if __name__ == "__main__":
    main()
