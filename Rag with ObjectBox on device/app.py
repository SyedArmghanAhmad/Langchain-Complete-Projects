import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

def inject_custom_css():
    """Inject custom CSS for improved visual design"""
    st.markdown("""
    <style>
        :root {
            --primary: #2A2B2E;
            --secondary: #5C6B73;
            --accent: #C6D8D3;
        }
        .main {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-family: 'Segoe UI', system-ui;
        }
        h1 {
            color: var(--primary);
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 0.5rem;
        }
        .sidebar .sidebar-content {
            background: var(--primary) !important;
            color: white !important;
        }
        .stButton>button {
            background: var(--secondary) !important;
            color: white !important;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(92,107,115,0.3);
        }
        .user-message {
            background: var(--secondary) !important;
            color: white;
            border-radius: 15px 15px 0 15px;
            margin: 1rem 0;
            max-width: 80%;
            float: right;
        }
        .bot-message {
            background: white !important;
            color: var(--primary);
            border-radius: 15px 15px 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
            max-width: 80%;
        }
        .references {
            background: #f8f9fa;
            border-left: 4px solid var(--accent);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_app():
    """Initialize application state and settings"""
    st.set_page_config(
        page_title="DocuMind - Intelligent Document Analysis",
        page_icon="📖",
        layout="wide"
    )
    inject_custom_css()

    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "processed" not in st.session_state:
        st.session_state.processed = False

def process_documents(upload_folder="./docs"):
    """Process uploaded documents with multi-document context"""
    try:
        os.makedirs(upload_folder, exist_ok=True)
        
        with st.spinner("🔍 Analyzing documents..."):
            loader = PyPDFDirectoryLoader(upload_folder)
            docs = loader.load()

            # Enhanced metadata for cross-document context
            for doc in docs:
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                doc.metadata['page'] += 1  # 1-based numbering

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=400,
                add_start_index=True,
                separators=["\n\n", "\n", "(?<=\. )", " "]
            )
            splits = text_splitter.split_documents(docs)

            # Add document relationships
            for i, split in enumerate(splits):
                if i > 0:
                    split.metadata['prev_doc'] = splits[i-1].metadata['source']
                if i < len(splits)-1:
                    split.metadata['next_doc'] = splits[i+1].metadata['source']

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create vector store with byte store
            os.makedirs("./byte_store", exist_ok=True)
            st.session_state.vectors = ObjectBox.from_documents(
                splits,
                embeddings,
                embedding_dimensions=768,
            )
            
            st.session_state.processed = True
            st.toast("Documents processed successfully!", icon="✅")

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processed = False

def build_retrieval_chain():
    """Build the retrieval chain with Mixtral"""
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=4000
    )

    prompt_template = ChatPromptTemplate.from_template("""
    Analyze documents collectively to answer the question. Consider:
    - Contextual relationships between documents
    - Temporal sequences and contradictions
    - Supporting evidence across sources

    Format response as:
    1. 🎯 Core Answer
    2. 🔗 Document Correlations
    3. 📌 Key References

    Documents:
    {context}

    Question: {input}

    Answer in markdown:
    """)

    return create_retrieval_chain(
        st.session_state.vectors.as_retriever(search_kwargs={"k": 5}),
        create_stuff_documents_chain(llm, prompt_template)
    )

def main():
    initialize_app()

    st.title("📖 DocuMind - Multi-Document Analysis")
    st.caption("Upload multiple PDFs and ask complex questions with cross-document understanding")

    with st.sidebar:
        st.header("Configuration")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple related PDFs for cross-document analysis"
        )

        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                # Save uploaded files
                upload_folder = "./docs"
                os.makedirs(upload_folder, exist_ok=True)
                for file in uploaded_files:
                    with open(os.path.join(upload_folder, file.name), "wb") as f:
                        f.write(file.getbuffer())
                process_documents(upload_folder)
            else:
                st.warning("Please upload documents first")

    if query := st.chat_input("Ask about your documents..."):
        if not st.session_state.processed:
            st.warning("Please process documents first")
            return

        # Display user message
        st.chat_message("user").markdown(
            f'<div class="user-message">{query}</div>', 
            unsafe_allow_html=True
        )

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                start_time = time.time()
                chain = build_retrieval_chain()
                response = chain.invoke({"input": query})

                # Format response
                answer = response["answer"]
                sources = {
                    (doc.metadata["source"], doc.metadata["page"])
                    for doc in response["context"]
                }

                # Display formatted answer
                st.markdown(answer)

                # Show references
                with st.expander("📚 Document References"):
                    for source, page in sorted(sources):
                        st.markdown(f"- **{source}** (page {page})")

                # Performance metrics
                st.caption(f"⏱️ Response generated in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()