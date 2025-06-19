import streamlit as st
import requests

st.set_page_config(page_title="Groq Translator", page_icon="🌍", layout="centered")

st.title("🌍 English to Hindi Translator")
st.caption("Built with FastAPI + Groq + LangChain + Streamlit 💬")

# Input box
user_input = st.text_area("Enter English text:", height=150, placeholder="e.g., How are you today?")

# Button to trigger translation
if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating via FastAPI & Groq..."):
            try:
                # Call your FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/translate",
                    json={"input": user_input}
                )
                if response.status_code == 200:
                    translated_text = response.json().get("response", "")
                    st.success("✅ Translated Text:")
                    st.text_area("Hindi Translation", translated_text, height=150)
                else:
                    st.error(f"Server Error: {response.status_code}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
