from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_setup import chain

app = FastAPI()

class Query(BaseModel):
    input: str

@app.get("/")
def root():
    return {"message": "LLM Translation API is running"}

@app.post("/translate")
async def translate_text(query: Query):
    try:
        print("\n--- Incoming Request ---")
        print("User Input:", query.input)

        response = chain.invoke({"input": query.input})

        print("Model Response:", response)
        print("--- End of Request ---\n")

        return {"response": response}
    except Exception as e:
        print("❌ Error occurred:", str(e))
        return {"error": str(e)}
