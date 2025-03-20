import streamlit as st
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

# Load environment variables
load_dotenv()

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MODEL = "mixtral-8x7b-32768"

# Initialize session states
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = []
    if "pdf_vector" not in st.session_state:
        st.session_state.pdf_vector = None
    if "web_vector" not in st.session_state:
        st.session_state.web_vector = None
    if "current_context" not in st.session_state:
        st.session_state.current_context = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

# Sidebar configuration
def configure_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Context selector
        input_type = st.radio("Choose context:", ["Website", "PDFs"])
        
        if input_type == "Website":
            url = st.text_input("🌐 Website URL", placeholder="Enter a website URL")
            if st.session_state.web_vector:
                st.caption(f"🌐 Current website: {st.session_state.current_context}")
        else:
            url = None
            new_uploads = st.file_uploader(
                "📁 Upload PDFs", type=["pdf"], 
                accept_multiple_files=True,
                help="Upload multiple PDF documents"
            )
            if new_uploads:
                existing_names = {f.name for f in st.session_state.uploaded_files}
                st.session_state.uploaded_files += [
                    f for f in new_uploads if f.name not in existing_names
                ]
            
            if st.session_state.uploaded_files:
                st.write("📚 Stored PDFs:")
                for f in st.session_state.uploaded_files:
                    st.caption(f"• {f.name}")
        
        chunk_size = st.slider("📏 Chunk Size", 500, 2000, DEFAULT_CHUNK_SIZE)
        chunk_overlap = st.slider("↔️ Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP)
        model_name = st.selectbox(
            "🧠 Model",
            ["mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0
        )
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
        
        return input_type, url, chunk_size, chunk_overlap, model_name

# Document processing
def process_documents(input_type, url, chunk_size, chunk_overlap):
    try:
        with st.status("Processing documents..."):
            embeddings = HuggingFaceEmbeddings()
            
            if input_type == "PDFs" and st.session_state.uploaded_files:
                new_files = [
                    f for f in st.session_state.uploaded_files 
                    if f.name not in st.session_state.processed_pdfs
                ]
                
                if new_files:
                    st.write("📄 Processing new PDF files...")
                    all_docs = []
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for file in new_files:
                            temp_filepath = os.path.join(temp_dir, file.name)
                            with open(temp_filepath, "wb") as f:
                                f.write(file.getbuffer())
                            loader = PyPDFLoader(temp_filepath)
                            docs = loader.load()
                            all_docs.extend(docs)
                            st.session_state.processed_pdfs.append(file.name)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    new_chunks = text_splitter.split_documents(all_docs)
                    
                    if st.session_state.pdf_vector:
                        st.write("🔄 Updating PDF vector store...")
                        st.session_state.pdf_vector.add_documents(new_chunks)
                    else:
                        st.write("🔮 Creating new PDF vector store...")
                        st.session_state.pdf_vector = FAISS.from_documents(new_chunks, embeddings)
                    
                    st.session_state.current_context = "PDFs"
                
            elif input_type == "Website" and url:
                if url != st.session_state.current_context:
                    st.write("🌐 Processing website content...")
                    loader = WebBaseLoader(url)
                    all_docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    new_chunks = text_splitter.split_documents(all_docs)
                    
                    st.write("🔮 Creating website vector store...")
                    st.session_state.web_vector = FAISS.from_documents(new_chunks, embeddings)
                    st.session_state.current_context = url
                
            st.success("Document processing complete!")
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.stop()

# Initialize LLM
def initialize_llm(model_name):
    groq_api_key = os.getenv('GROQ_API_KEY')
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Create retrieval chain
def create_chain(llm, input_type):
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}
        
        Question: {input}"""
    )
    
    if input_type == "PDFs" and st.session_state.pdf_vector:
        retriever = st.session_state.pdf_vector.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        return create_retrieval_chain(retriever, document_chain)
    elif input_type == "Website" and st.session_state.web_vector:
        retriever = st.session_state.web_vector.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        return create_retrieval_chain(retriever, document_chain)
    return None

# Display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("context"):
                st.caption(f"📚 Context: {message['context']}")

# Handle user input
def handle_user_input(prompt, input_type, retrieval_chain):
    if not retrieval_chain:
        st.error("Please process documents first!")
        st.stop()
        
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "context": input_type
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🔍 Querying: {input_type}")

    try:
        with st.spinner("💭 Thinking..."):
            response = retrieval_chain.invoke({"input": prompt})
            answer = response['answer']
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context": input_type
        })
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"📚 Source: {input_type}")
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Main function
def main():
    st.title("📚 Context-Aware Document Assistant")
    st.caption("RAG with separate document contexts and smart switching")
    
    initialize_session_state()
    input_type, url, chunk_size, chunk_overlap, model_name = configure_sidebar()
    
    # Process documents when needed
    process_trigger = False
    if input_type == "PDFs" and st.session_state.uploaded_files:
        process_trigger = any(f.name not in st.session_state.processed_pdfs 
                            for f in st.session_state.uploaded_files)
    elif input_type == "Website" and url:
        process_trigger = url != st.session_state.current_context

    if process_trigger:
        process_documents(input_type, url, chunk_size, chunk_overlap)
    
    llm = initialize_llm(model_name)
    retrieval_chain = create_chain(llm, input_type)
    
    display_chat_messages()
    
    if prompt := st.chat_input("Ask anything about the documents..."):
        handle_user_input(prompt, input_type, retrieval_chain)

# Run the app
if __name__ == "__main__":
    main()