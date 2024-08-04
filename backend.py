from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import re

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_input: str

groq_api_key = os.getenv('groq_api_key')

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable assistant.
You will answer the user's questions and assist them with various tasks.
User input: {user_input}
""")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

def format_to_html(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)  # Italic
    text = text.replace('\n', '<br>')  # New lines to <br>
    return text

def get_response(user_input):
    prompt = prompt_template.format(user_input=user_input)
    response = llm.invoke(prompt)
    response_text = response.content  # Extract the content from the AIMessage object
    print("Generated response:", response_text)  # Print response for debugging
    formatted_response = format_to_html(response_text)  # Format response to HTML
    return formatted_response

@app.post("/query")
async def query(request: QueryRequest):
    response = get_response(request.user_input)
    return {"response": response}

# Mount the frontend directory to serve static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

