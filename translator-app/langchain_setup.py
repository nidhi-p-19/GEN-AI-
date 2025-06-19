from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Debug print
print("🔑 Loaded API Key:", api_key[:6] + "..." if api_key else "❌ Not found")

llm = ChatGroq(
    model="gemma2-9b-it",  # Change to gemma-9b-it if that’s what you’re using
    api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to Hindi."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
