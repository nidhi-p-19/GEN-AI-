import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# --- Imports for Embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings
# --- Imports for Vector Store ---
from langchain_community.vectorstores import FAISS

# --- Imports for LLM (Groq) ---
from langchain_groq import ChatGroq

# --- Other Langchain and Standard Imports ---
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

import os
from typing import List, Dict, Any

# --- Document Processing Functions ---
# (Assuming these are working fine based on previous terminal output)
def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        print("No PDF documents provided.")
        return text

    print(f"Starting text extraction from {len(pdf_docs)} PDF(s)...")
    for i, pdf in enumerate(pdf_docs):
        print(f"Processing PDF {i+1}/{len(pdf_docs)}: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            num_pages = len(pdf_reader.pages)
            print(f"  Found {num_pages} pages.")
            for page_num, page in enumerate(pdf_reader.pages):
                if (page_num + 1) % 10 == 0:
                    print(f"    Extracting text from page {page_num+1}/{num_pages}...")
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception as e:
                    print(f"    Error extracting text from page {page_num+1}: {e}")
            print(f"Finished processing PDF {i+1}: {pdf.name}")
        except Exception as e:
            print(f"Error reading PDF file {pdf.name}: {e}")
    print(f"Total text extracted: {len(text)} characters.")
    return text

def get_text_chunks(text):
    if not text:
        print("No text to chunk.")
        return []

    print("Starting text chunking...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Text chunking complete. Created {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        print("No text chunks to create vector store.")
        return None

    print(f"Creating vector store with {len(text_chunks)} chunks...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

from langchain.chains import LLMChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vectorstore):
    if vectorstore is None:
        print("Cannot create conversation chain: vectorstore is None.")
        return None

    print("Fixing the output_key issue with manual chain setup...")

    try:
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-8b-8192",
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # ✅ Explicit fix: Set return_source_documents=False temporarily OR do manual chaining.
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            output_key="answer",  # Important!
            return_source_documents=False  # Try disabling this to remove dual outputs.
        )

        print("Fixed chain initialized successfully.")
        return conversation_chain

    except Exception as e:
        print(f"Error initializing conversation chain: {e}")
        return None


def handle_userinput(user_question):
    """Handles user input and displays conversation history."""
    if st.session_state.conversation is None:
        st.warning("LLM is not initialized. Please process your documents first.")
        return

    print(f"User question: {user_question}")
    try:
        # The invoke method expects a dictionary with the input key.
        # For ConversationalRetrievalChain, this is usually 'question'.
        response = st.session_state.conversation.invoke({'question': user_question})

        # --- DEBUGGING START ---
        print(f"Raw response from chain.invoke: {response}")
        if isinstance(response, dict):
            print(f"Response keys: {response.keys()}")
            if 'answer' in response:
                print(f"Answer content found. Type: {type(response['answer'])}")
            else:
                print("WARNING: 'answer' key not found in the response.")
            if 'chat_history' in response: # Check if the response itself contains history
                print(f"'chat_history' found in response. Length: {len(response['chat_history'])}")
            else:
                print("WARNING: 'chat_history' key not found in the response dict.")
        else:
            print(f"Response is not a dictionary. It's of type: {type(response)}")
        # --- DEBUGGING END ---

        # Safely get the answer.
        answer = response.get('answer', 'Sorry, I could not generate an answer.') if isinstance(response, dict) else 'Sorry, I could not generate an answer.'

        # Display the bot's answer
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

        # --- CRITICAL CHANGE FOR HISTORY MANAGEMENT ---
        # Instead of trying to parse 'chat_history' from the response directly,
        # we rely on the memory object attached to the conversation chain.
        # When the chain runs, it updates its memory.
        # We can then read the messages from the memory for display.
        
        # Make sure the conversation and memory are valid before accessing
        if st.session_state.conversation and hasattr(st.session_state.conversation, 'memory'):
            # Access the messages from memory and update st.session_state.chat_history
            # This ensures we are working with the chain's actual history.
            # The format of messages might be AIMessage, HumanMessage, etc.
            # For display, we need to extract content and type.
            
            # Check if memory has messages and if they are in the expected format
            if hasattr(st.session_state.conversation.memory.chat_memory, 'messages'):
                updated_chat_history = st.session_state.conversation.memory.chat_memory.messages
                
                # Reconstruct the history for display if needed
                # Clear current display history to avoid duplicates, then repopulate
                # st.session_state.chat_history = updated_chat_history # This might be too naive for re-renders

                # A more robust way is to update the session state for future rendering.
                # The loop in main() will use this updated list.
                st.session_state.chat_history = updated_chat_history
                
            else:
                print("WARNING: Memory does not contain '.messages' attribute.")
        else:
            print("WARNING: Conversation or memory not properly initialized to access history.")


    except Exception as e:
        st.error(f"An error occurred during conversation: {e}")
        print(f"Error in handle_userinput: {e}")


def main():
    """Main function to run the Streamlit app."""
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Initialize as an empty list for displaying messages
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        st.session_state.pdf_docs = uploaded_files

        if st.button("Process", key="process_button"):
            if not st.session_state.pdf_docs:
                st.warning("Please upload some PDF documents first.")
            else:
                with st.spinner("Processing documents..."):
                    print("Starting document processing flow...")
                    raw_text = get_pdf_text(st.session_state.pdf_docs)
                    if not raw_text:
                        st.error("Failed to extract text from PDFs. Please check PDF content or format.")
                        st.session_state.conversation = None
                        return

                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Failed to chunk text. The document might be too short or improperly formatted.")
                        st.session_state.conversation = None
                        return

                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore is None:
                        st.error("Failed to create vector store. Check for errors in vector store creation.")
                        st.session_state.conversation = None
                        return

                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.success("Documents processed and ready for chat!")
                        st.session_state.chat_history = [] # Clear history for new session
                    else:
                        st.error("Failed to initialize the conversation chain. Check LLM setup and API keys.")

    # Chat interface section
    # Display existing chat history first
    # This loop ensures that all previous messages are rendered on the screen.
    # It iterates through the chat_history stored in session_state.
    for message in st.session_state.chat_history:
        if message.type == "human":
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif message.type == "ai":
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else: # Fallback for unknown types or direct strings if they somehow get in history
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        # Display user's message immediately
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

        # Now process the input and get the bot's response
        handle_userinput(user_question)
        
        # After handle_userinput, the bot's answer is displayed.
        # The chat history should be updated in st.session_state.chat_history by handle_userinput.
        # Streamlit will rerun and the loop above will display the updated history.

if __name__ == '__main__':
    main()
