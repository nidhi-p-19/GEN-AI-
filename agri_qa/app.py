import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os
import requests # For fetching web content
from bs4 import BeautifulSoup # For parsing HTML

# --- Configuration ---
DATA_SOURCE_URL = 'paste ur link'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GENERATIVE_MODEL_NAME = 'google/flan-t5-base'
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 5

# --- Data Loading and Preprocessing ---

def extract_text_from_url(url):
    """Fetches and extracts text from a webpage."""
    print(f"Attempting to fetch content from URL: {url}")
    try:
        response = requests.get(url, timeout=10) # Added timeout for robustness
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Extracting text from specific HTML elements ---
        # This is often the trickiest part as website structures vary.
        # We'll try to get text from main content areas, paragraphs, and headings.
        # You might need to inspect the webpage's HTML source (Ctrl+U or right-click -> Inspect)
        # to find the best selectors for the relevant information.

        # Common selectors to try:
        # main content areas: article, main, div.content, div.post-content
        # paragraphs: p
        # headings: h1, h2, h3, h4

        # Let's try to extract text from paragraphs and potentially headings
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4'])
        extracted_text = ' '.join([element.get_text() for element in text_elements])

        print(f"Successfully fetched and extracted text. Total characters: {len(extracted_text)}")
        return extracted_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return f"Error fetching URL: {e}"
    except Exception as e:
        print(f"Error parsing HTML from {url}: {e}")
        return f"Error parsing HTML: {e}"

def clean_text(text):
    """Cleans extracted text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace with single space
    # Keep alphanumeric characters, spaces, periods, commas, hyphens, slashes, parentheses.
    # This regex removes most special characters that might not be useful for embeddings.
    text = re.sub(r'[^\w\s\.\,\-\/\(\)]', '', text)
    text = text.strip()
    print(f"Cleaned text length: {len(text)}")
    return text

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunks text into smaller pieces with overlap."""
    if not text:
        return []
    words = text.split()
    chunks = []
    if len(words) <= size:
        return [" ".join(words)]

    start = 0
    while start < len(words):
        end = min(start + size, len(words)) # Ensure end doesn't go past the last word
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += size - overlap # Move start for overlap
        if start >= len(words):
            break
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- Initialize Models ---
print("Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    raise

print(f"Loading generative model '{GENERATIVE_MODEL_NAME}'...")
try:
    qa_pipeline = pipeline("text2text-generation", model=GENERATIVE_MODEL_NAME)
    print("Generative model loaded successfully.")
except Exception as e:
    print(f"Error loading generative model: {e}")
    raise

# --- Indexing ---
INDEX_FILE = "faiss_index.idx"
CHUNKS_FILE = "chunks.npy"

def create_or_load_index():
    """Fetches content from URL, creates FAISS index and embeddings if they don't exist, otherwise loads them."""
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index and chunks...")
        try:
            index = faiss.read_index(INDEX_FILE)
            all_chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
            print(f"Index loaded with {index.ntotal} vectors.")
            return index, all_chunks
        except Exception as e:
            print(f"Error loading index/chunks: {e}. Recreating index.")
            pass # Will proceed to create new index

    print("Creating new FAISS index and embeddings from URL...")
    # Use the new function for URL fetching
    extracted_text = extract_text_from_url(DATA_SOURCE_URL)

    if isinstance(extracted_text, str) and extracted_text.startswith("Error:"):
        print(f"Fatal Error during data fetching/processing: {extracted_text}")
        return None, None

    if not extracted_text:
        print("Error: No text could be extracted from the source.")
        return None, None

    cleaned_text = clean_text(extracted_text)
    if not cleaned_text:
        print("Error: Cleaned text is empty. Check source content or cleaning process.")
        return None, None

    all_chunks = chunk_text(cleaned_text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not all_chunks:
        print("Error: No chunks generated. Check source content or chunking logic.")
        return None, None

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    try:
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
        print(f"Embeddings generated. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None, None

    # Build FAISS index
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension: {dimension}")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    print(f"Index created with {index.ntotal} vectors.")

    # Save index and chunks
    try:
        faiss.write_index(index, INDEX_FILE)
        np.save(CHUNKS_FILE, np.array(all_chunks, dtype=object))
        print(f"Index saved to {INDEX_FILE}.")
        print(f"Chunks saved to {CHUNKS_FILE}.")
    except Exception as e:
        print(f"Error saving index or chunks: {e}")

    return index, all_chunks

# --- Querying ---
def query_system(question, index, all_chunks, top_k=TOP_K):
    """Performs similarity search and generates an answer."""
    if not index or not all_chunks:
        print("Error: Index or chunks are not available.")
        return "Error: System is not initialized correctly. Please check logs."

    try:
        # Embed the question
        question_embedding = embedding_model.encode([question])
        print(f"Question embedding shape: {question_embedding.shape}")

        # Perform similarity search
        distances, indices = index.search(np.array(question_embedding).astype('float32'), top_k)
        print(f"Raw FAISS search results: indices={indices}, distances={distances}")

        relevant_chunk_texts = []
        valid_indices = indices[0]

        for idx in valid_indices:
            if idx != -1 and idx < len(all_chunks):
                relevant_chunk_texts.append(all_chunks[idx])

        print(f"Successfully retrieved {len(relevant_chunk_texts)} valid relevant chunks.")
        # print("Retrieved chunk texts:", relevant_chunk_texts) # Uncomment for detailed debugging of chunk content

        if not relevant_chunk_texts:
            print("No relevant chunks found for the question.")
            context_text = "No specific information found in the documents for this query."
        else:
            context_text = "\n\n---\n\n".join(relevant_chunk_texts)

        prompt = (
            "You are an expert agricultural advisor. Your task is to extract information about cotton cultivation from the provided guidelines. "
            "Answer the user's question directly and concisely, using only the information from the provided context. "
            "If the information is not present, state that clearly.\n\n"
            "--- Agricultural Guidelines ---\n"
            f"{context_text}\n"
            "-----------------------------\n\n"
            f"User Question: {question}\n\n"
            "Expert Answer:"
        )

        print(f"Generating answer for: '{question}'...")
        result = qa_pipeline(prompt, max_length=250, num_beams=4, early_stopping=True)

        if result and isinstance(result, list) and 'generated_text' in result[0]:
            answer = result[0]['generated_text'].strip()
            print(f"Generated Answer: '{answer}'")
            return answer
        else:
            print("Error: Pipeline returned an unexpected result format.")
            return "An error occurred during answer generation."

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        return f"An unexpected error occurred: {e}"

# --- Gradio UI ---

# Create index and load data once when the script starts
print("Initializing Q&A system...")
faiss_index, loaded_chunks = create_or_load_index()

# Define the Gradio Interface function
def agricultural_qa_interface(question):
    if not question.strip():
        return "Please ask a question."
    return query_system(question, faiss_index, loaded_chunks)

# Gradio Interface Setup
theme = gr.themes.Soft()

with gr.Blocks(theme=theme, title="Agri-Helper: Q&A") as demo:
    gr.Markdown("# 🌾 Agri-Helper: Crop Management Q&A 🌾")
    gr.Markdown("Ask questions about cultivation, pests, diseases, and management practices based on TNAU agricultural guidelines.")

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the symptoms of root rot in cotton?",
                lines=2,
                max_lines=3
            )
            submit_button = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Answer",
                placeholder="The answer will appear here...",
                interactive=False,
                lines=5,
                max_lines=10
            )

    gr.Markdown("---")
    gr.Markdown("### Example Questions:")
    gr.Examples(
        [
            ["What are the ideal soil conditions for cotton cultivation?"],
            ["What are the main pests of cotton in India?"],
            ["How to manage aphids in cotton?"],
            ["What are the symptoms of leaf spot in cotton?"],
            ["When is the best time to sow cotton?"],
            ["How to control bollworms?"],
            ["What are the signs of nutrient deficiency in cotton?"],
            ["Describe the effects of water scarcity on cotton."],
        ],
        inputs=question_input,
        label="Try these questions"
    )

    submit_button.click(
        agricultural_qa_interface,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    if faiss_index is None:
        print("\n--------------------------------------------------------------")
        print("System initialization failed. Please check the console for")
        print("specific errors related to PDF processing, model loading, or indexing.")
        print("Ensure the PDF file exists and is readable, and models can be downloaded.")
        print("--------------------------------------------------------------")
    else:
        print("\n--------------------------------------------------------------")
        print("Agri-Helper Q&A System is ready!")
        print("Access the UI at the URL provided by Gradio (usually http://127.0.0.1:7860).")
        print("--------------------------------------------------------------")
        demo.launch(debug=True)
