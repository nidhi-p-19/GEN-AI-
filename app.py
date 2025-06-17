import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Set Streamlit title
st.title("Chat with Groq + Ollama + FAISS")

# Run the vector setup once
if "vector" not in st.session_state:
    # Step 1: Load the document
    loader = WebBaseLoader("https://cloud.google.com/discover/what-are-ai-agents")
    docs = loader.load()

    # Step 2: Split the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Step 3: Embed & store in FAISS
    embeddings = OllamaEmbeddings()
    vector = FAISS.from_documents(split_docs, embeddings)

    # Save in session state
    st.session_state.vector = vector
    st.session_state.docs = docs

# Load the Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192")

# Define prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the question as accurately as possible using the provided context only.

<context>
{context}
</context>

Question: {input}
""")

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Prompt input
prompt = st.text_input("Ask your question:")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    end = time.process_time()
    
    st.subheader("Answer:")
    st.write(response['answer'])
    st.caption(f"⏱ Response time: {round(end - start, 2)} seconds")

    with st.expander("Document Similarity Matches"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Match {i+1}:**")
            st.write(doc.page_content)
            st.write("---")
