# 🌐 English to Hindi Translator — FastAPI + LangChain + Groq + Streamlit

A full-stack LLM-powered translator app that converts English text into Hindi using Groq’s blazing-fast models and LangChain pipelines — wrapped in a clean FastAPI backend and Streamlit frontend.

---

## 🔧 Tech Stack

- 🧠 **Groq LLM** (e.g., `gemma2-9b-it`)
- 🔗 **LangChain** for chaining prompts, models, and parsers
- ⚙️ **FastAPI** as the backend to serve model predictions
- 🎨 **Streamlit** frontend for user interaction
- 🔐 **.env** with API key management

---

## 🚀 How It Works

1. User inputs English text in the frontend
2. Streamlit sends request to FastAPI endpoint `/translate`
3. FastAPI sends the input to Groq model through a LangChain chain
4. Model returns the translated Hindi response
5. Response is shown beautifully in the frontend UI

---

## 💻 Run Locally

