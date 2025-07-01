from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
# You can change these models based on your hardware and preference.
# For Q&A, a Seq2Seq model like T5 or BART works well.
# For embeddings, sentence-transformers are efficient.

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# CHOOSE YOUR QA MODEL:
# - "google/flan-t5-large" (good general purpose Q&A, ~770MB model, ~1.5GB VRAM/RAM needed)
# - "facebook/bart-large-cnn" (better for summarization, but can do Q&A)
# - "google/flan-t5-base" (smaller, faster, but less capable)
# - "t5-small" (very small, fast, but limited Q&A quality)
QA_MODEL_NAME = "google/flan-t5-base"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Embeddings ---
def get_embedding_model():
    """Initializes and returns the sentence-transformers embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return embeddings

# --- QA Model ---
def get_qa_pipeline():
    """Initializes and returns the Hugging Face pipeline for QA."""
    print(f"Loading QA model: {QA_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_NAME)

    # Create a Hugging Face pipeline
    # For QA, the task can be 'text2text-generation' or simply 'question-answering' if the model supports it.
    # FLAN-T5 is a text-to-text model.
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # Adjust max_length as needed for your answers
        temperature=0.3, # Lower temperature for more factual answers
        top_p=0.95,
        repetition_penalty=1.15
    )
    print("QA model loaded.")
    return qa_pipeline

# --- RAG System ---
class YouTubeQASystem:
    def __init__(self, qa_pipeline, embeddings):
        self.qa_pipeline = qa_pipeline
        self.embeddings = embeddings
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    def load_and_index_transcript(self, transcript_text):
        """
        Splits the transcript, creates embeddings, and builds a FAISS vector store.
        """
        print("Splitting transcript into chunks...")
        # LangChain's TextSplitter returns Document objects
        docs = self.text_splitter.create_documents([transcript_text])
        print(f"Created {len(docs)} chunks.")

        print("Creating embeddings and building vector store...")
        # FAISS needs documents with metadata if we want to embed them
        # For simpler usage, we can embed the content directly
        # However, to properly use LangChain's DocumentLoader, we should treat `docs` as loaded documents
        # If you want to add metadata to your chunks (like start/end times), you'd map them here.
        # For now, we'll embed the page_content of each document.

        # Create FAISS index from documents
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        print("Vector store built and retriever created.")

    def setup_qa_chain(self):
        """Sets up the RetrievalQA chain."""
        if not self.retriever:
            raise ValueError("Vector store and retriever not initialized. Load transcript first.")

        # Customize the prompt for better Q&A
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Be concise and direct in your answers.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""
        QA_PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Use the HuggingFacePipeline for the LLM
        llm_chain = HuggingFacePipeline(pipeline=self.qa_pipeline)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm_chain,
            chain_type="map_reduce", # 'stuff' puts all relevant documents into one prompt
            retriever=self.retriever,
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True # Optional: to see which parts of the transcript were used
        )
        print("QA chain set up.")

    def answer_question(self, question):
        """
        Asks a question and returns the answer using the RAG system.
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Load transcript and set up the chain first.")

        print(f"Answering question: '{question}'")
        result = self.qa_chain({"query": question})
        answer = result["result"]
        print(f"Raw answer: {answer}")

        # Post-processing the answer if needed
        # For FLAN-T5, sometimes it might include leading/trailing spaces or repetitions.
        cleaned_answer = answer.strip()

        # Optionally, you can try to extract sources if return_source_documents=True
        # sources = [doc.metadata for doc in result.get("source_documents", [])]
        # For our current chunking, metadata is just {'start_index': ...}, not directly useful as sources without more complex handling.

        return cleaned_answer
