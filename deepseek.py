import streamlit as st
import os
from dotenv import load_dotenv
import time

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Streamlit App UI
st.title("DeepSeek Chatbot ")

# Initialize only once
if "vector" not in st.session_state:
    st.write("Loading & indexing content...")

    loader = WebBaseLoader("https://cloud.google.com/learn/what-is-artificial-intelligence")  # You can change this URL
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)[:50]

    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vector = FAISS.from_documents(split_docs, embeddings)

    st.session_state.vector = vector
    st.session_state.docs = docs

# Load Ollama LLM (DeepSeek model must be pulled via ollama)
llm = ChatOllama(model="deepseek-r1")  # assumes `ollama pull deepseek` was done

# Create prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
<context>
{context}
</context>

Question: {input}
""")

# Create chains
retriever = st.session_state.vector.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
prompt = st.text_input("Ask your question:")

if prompt:
    start = time.time()
    response = retrieval_chain.invoke({"input": prompt})
    end = time.time()

    st.subheader("💬 Response:")
    st.write(response['answer'])
    st.caption(f"⏱️ Response time: {round(end - start, 2)} seconds")

    with st.expander("🔍 Context Matches"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Match {i+1}:**")
            st.write(doc.page_content)
            st.write("---")
