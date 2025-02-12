from fastapi import FastAPI  # 'FastAPI' instead of 'FastApi'
import uvicorn
import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="LangChain Chatbot API",
    version="1.0",
    description="A simple API for interacting with a LangChain Chatbot",
)

# Add GroQ routes
# Define Groq-based chatbot
llm = ChatGroq(model_name="mixtral-8x7b-32768")

add_routes(
    app,
    ChatGroq(),
    path="/groq"
    )

prompt=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")

add_routes(
    app,
    prompt|llm,
    path="/essay"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)


