import gradio as gr
from youtube_utils import fetch_youtube_transcript
from llm_qa_system import YouTubeQASystem, get_qa_pipeline, get_embedding_model
import os
import torch

# --- Global Variables ---
# Initialize LLM and Embeddings once
try:
    qa_pipeline = get_qa_pipeline()
    embeddings = get_embedding_model()
    qa_system = YouTubeQASystem(qa_pipeline, embeddings)
    is_system_ready = True
    print("System initialized successfully.")
except Exception as e:
    print(f"Error initializing system: {e}")
    is_system_ready = False
    qa_system = None # Ensure it's None if initialization fails

# Check for GPU availability
# You can uncomment this and try to force CUDA if you have a compatible GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set to your GPU index if needed
# if torch.cuda.is_available():
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU. For better performance, consider a GPU.")


# --- Gradio Interface Functions ---

def process_video_and_query(video_url, question):
    """
    1. Fetches transcript.
    2. Indexes transcript for RAG.
    3. Sets up the QA chain.
    4. Answers the question.
    """
    if not is_system_ready or qa_system is None:
        return "System not initialized properly. Please check logs.", gr.update(visible=False)

    if not video_url or not question:
        return "Please provide both a YouTube URL and a question.", gr.update(visible=False)

    try:
        # 1. Fetch Transcript
        print(f"Fetching transcript for: {video_url}")
        transcript_text, _ = fetch_youtube_transcript(video_url) # We only need the text for this example
        if not transcript_text:
            return "Could not fetch transcript for the provided URL.", gr.update(visible=False)
        print(f"Transcript fetched. Length: {len(transcript_text)} characters.")

        # 2. Load and Index Transcript
        qa_system.load_and_index_transcript(transcript_text)

        # 3. Setup QA Chain
        qa_system.setup_qa_chain()

        # 4. Answer Question
        answer = qa_system.answer_question(question)

        return answer, gr.update(visible=True) # Show the answer output

    except ValueError as ve:
        return f"Input error: {ve}", gr.update(visible=False)
    except RuntimeError as re:
        return f"Runtime error: {re}", gr.update(visible=False)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}", gr.update(visible=False)

def greet(video_url, question):
    """
    This function is called when the 'Submit' button is clicked.
    It orchestrates the video processing and question answering.
    """
    answer, answer_state = process_video_and_query(video_url, question)
    return answer, answer_state

# --- Gradio App Definition ---

# Define CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    background-color: #f4f7f6; /* Light grey background */
    padding: 20px;
    border-radius: 10px;
}
h1, h2 {
    color: #2c3e50; /* Dark blue/grey */
    text-align: center;
    margin-bottom: 20px;
}
.gradio-input {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 12px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.gradio-button {
    background-color: #3498db; /* Blue button */
    color: white;
    font-weight: bold;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gradio-button:hover {
    background-color: #2980b9; /* Darker blue on hover */
}
.error-message {
    color: #e74c3c; /* Red for errors */
    font-weight: bold;
}
.output-box {
    border: 1px solid #bdc3c7; /* Light grey border for output */
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    min-height: 100px;
    text-align: left;
    color: #34495e; /* Dark text */
}
"""

with gr.Blocks(css=custom_css, title="YouTube Q&A Bot") as demo:
    gr.HTML("""
        <div style="text-align: center;">
            <h1>🎓 YouTube Video Q&A Bot</h1>
            <p>Ask questions about YouTube videos by providing their URL and your query.</p>
            <p>Powered by open-source LLMs and RAG!</p>
        </div>
    """)

    with gr.Row():
        video_url_input = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter a YouTube video URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)",
            lines=1,
            elem_classes=["gradio-input"]
        )
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="What is this video about?",
            lines=3,
            elem_classes=["gradio-input"]
        )

    submit_button = gr.Button("Ask", elem_classes=["gradio-button"])

    # Output for the answer
    answer_output = gr.Textbox(
        label="Answer",
        value="Please enter a YouTube URL and your question, then click 'Ask'.",
        interactive=False, # User cannot edit this
        visible=False, # Initially hidden
        elem_classes=["output-box"]
    )

    # Handle the button click event
    submit_button.click(
        fn=greet,
        inputs=[video_url_input, question_input],
        outputs=[answer_output, answer_output] # Output goes to the same textbox, but we control visibility
    )

    # Display system status
    if not is_system_ready:
        gr.Markdown("<h2 class='error-message'>Warning: System initialization failed. Please check logs for details.</h2>")
    else:
        gr.Markdown("<p style='text-align: center; color: #555;'>System ready. Models loaded.</p>")


if __name__ == "__main__":
    # For production, consider setting api_key for YouTube API if it's frequently used
    # os.environ["YOUTUBE_API_KEY"] = "YOUR_API_KEY_HERE" # Or load from .env

    # Launch the Gradio app
    # Share=True creates a public link (temporary) if you want to share it.
    # For local use, you can omit `share=True`.
    demo.launch() # Use demo.launch(share=True) to get a shareable link
