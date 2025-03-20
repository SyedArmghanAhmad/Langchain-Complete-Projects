from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import os
import tempfile
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Context-Aware Document Assistant", version="1.0")

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MODEL = "mixtral-8x7b-32768"

# Pydantic models for request/response
class DocumentInput(BaseModel):
    input_type: str  # "Website" or "PDFs"
    url: Optional[str] = None
    files: Optional[List[UploadFile]] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    model_name: str = DEFAULT_MODEL

class QueryInput(BaseModel):
    prompt: str

# Global state (for simplicity; replace with a proper database in production)
state = {
    "messages": [],
    "processed_pdfs": [],
    "pdf_vector": None,
    "web_vector": None,
    "current_context": None,
    "uploaded_files": []
}

# Initialize LLM
def initialize_llm(model_name: str):
    groq_api_key = os.getenv('GROQ_API_KEY')
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Process documents
def process_documents(input_type: str, url: Optional[str], files: Optional[List[UploadFile]], chunk_size: int, chunk_overlap: int):
    try:
        embeddings = HuggingFaceEmbeddings()
        
        if input_type == "PDFs" and files:
            new_files = [f for f in files if f.filename not in state["processed_pdfs"]]
            
            if new_files:
                all_docs = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in new_files:
                        temp_filepath = os.path.join(temp_dir, file.filename)
                        with open(temp_filepath, "wb") as f:
                            f.write(file.file.read())
                        loader = PyPDFLoader(temp_filepath)
                        docs = loader.load()
                        all_docs.extend(docs)
                        state["processed_pdfs"].append(file.filename)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                new_chunks = text_splitter.split_documents(all_docs)
                
                if state["pdf_vector"]:
                    state["pdf_vector"].add_documents(new_chunks)
                else:
                    state["pdf_vector"] = FAISS.from_documents(new_chunks, embeddings)
                
                state["current_context"] = "PDFs"
                
        elif input_type == "Website" and url:
            if url != state["current_context"]:
                loader = WebBaseLoader(url)
                all_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                new_chunks = text_splitter.split_documents(all_docs)
                
                state["web_vector"] = FAISS.from_documents(new_chunks, embeddings)
                state["current_context"] = url
                
        return {"status": "success", "message": "Documents processed successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

# Create retrieval chain
def create_chain(llm, input_type: str):
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}
        
        Question: {input}"""
    )
    
    if input_type == "PDFs" and state["pdf_vector"]:
        retriever = state["pdf_vector"].as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        return create_retrieval_chain(retriever, document_chain)
    elif input_type == "Website" and state["web_vector"]:
        retriever = state["web_vector"].as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        return create_retrieval_chain(retriever, document_chain)
    return None

# FastAPI endpoints
@app.post("/process-documents")
async def process_documents_endpoint(input_data: DocumentInput):
    result = process_documents(
        input_data.input_type,
        input_data.url,
        input_data.files,
        input_data.chunk_size,
        input_data.chunk_overlap
    )
    return JSONResponse(content=result)

@app.post("/query")
async def query_endpoint(query: QueryInput):
    llm = initialize_llm(DEFAULT_MODEL)
    retrieval_chain = create_chain(llm, state["current_context"])
    
    if not retrieval_chain:
        raise HTTPException(status_code=400, detail="Please process documents first!")
    
    response = retrieval_chain.invoke({"input": query.prompt})
    answer = response['answer']
    
    state["messages"].append({"role": "user", "content": query.prompt})
    state["messages"].append({"role": "assistant", "content": answer})
    
    return JSONResponse(content={"answer": answer})

@app.get("/chat-history")
async def get_chat_history():
    return JSONResponse(content=state["messages"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}