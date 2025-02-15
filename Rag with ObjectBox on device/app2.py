import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

def inject_custom_css():
    """Inject modern, minimal CSS styling"""
    st.markdown("""
    <style>
        :root {
            --primary: #2B2D42;
            --secondary: #8D99AE;
            --accent: #EF233C;
            --light: #EDF2F4;
        }
        
        html, body, [class*="css"]  {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        
        .main {
            background: var(--light);
        }
        
        h1 {
            color: var(--primary);
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 1.5rem;
        }
        
        .stSidebar {
            background: var(--primary) !important;
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        
        .sidebar .block-container {
            padding-top: 2rem;
        }
        
        .uploaded-file {
            padding: 0.5rem;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            margin: 0.5rem 0;
            font-size: 0.9em;
        }
        
        .stButton>button {
            background: var(--accent) !important;
            color: white !important;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            transition: all 0.2s ease;
            border: none;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(239,35,60,0.25);
        }
        
        .chat-message {
            max-width: 80%;
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 12px;
            animation: fadeIn 0.3s ease;
        }
        
        .user-message {
            background: #EF233C;
            color: white;
            border-radius: 15px 15px 0 15px;
            padding: 1rem;
            margin: 1rem 0;
            max-width: 80%;
            float: right;
        }
        
        .bot-message {
            background: white;
            color: #2B2D42;
            border-radius: 15px 15px 15px 0;
            padding: 1rem;
            margin: 1rem 0;
            max-width: 80%;
            border: 1px solid #EDF2F4;
        }
        
        .references {
            background: rgba(141,153,174,0.08);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            border-left: 3px solid var(--secondary);
        }
        
        .performance-metric {
            color: var(--secondary);
            font-size: 0.85em;
            margin-top: 1rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_app():
    """Initialize application state and settings"""
    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="📄",
        layout="centered"
    )
    inject_custom_css()
    
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "processed" not in st.session_state:
        st.session_state.processed = False

def process_documents(upload_folder="./docs"):
    """Process uploaded documents (unchanged from previous version)"""
    try:
        os.makedirs(upload_folder, exist_ok=True)
        
        with st.spinner("🔍 Analyzing documents..."):
            loader = PyPDFDirectoryLoader(upload_folder)
            docs = loader.load()

            for doc in docs:
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                doc.metadata['page'] += 1

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=400,
                add_start_index=True,
                separators=["\n\n", "\n", "(?<=\. )", " "]
            )
            splits = text_splitter.split_documents(docs)

            for i, split in enumerate(splits):
                if i > 0:
                    split.metadata['prev_doc'] = splits[i-1].metadata['source']
                if i < len(splits)-1:
                    split.metadata['next_doc'] = splits[i+1].metadata['source']

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

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
    """Build the retrieval chain with Mixtral (unchanged)"""
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
    
    # Header Section
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("DocuMind AI")
        st.caption("Intelligent Document Analysis with Cross-Reference Capabilities")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/7858/7858975.png", width=80)

    # Sidebar Configuration
    with st.sidebar:
        st.header("Document Hub")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple documents for cross-analysis"
        )
        
        if uploaded_files:
            st.subheader("Uploaded Files")
            for file in uploaded_files:
                st.markdown(f'<div class="uploaded-file">{file.name}</div>', 
                           unsafe_allow_html=True)
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                upload_folder = "./docs"
                os.makedirs(upload_folder, exist_ok=True)
                for file in uploaded_files:
                    with open(os.path.join(upload_folder, file.name), "wb") as f:
                        f.write(file.getbuffer())
                process_documents(upload_folder)
            else:
                st.warning("Please upload documents first")

    # Chat Interface
    if query := st.chat_input("Ask about your documents..."):
        if not st.session_state.processed:
            st.warning("Please process documents first")
            return

        # User Message
        st.chat_message("user").markdown(
            f'<div class="chat-message user-message">{query}</div>',
            unsafe_allow_html=True
        )

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                start_time = time.time()
                chain = build_retrieval_chain()
                response = chain.invoke({"input": query})
                answer = response["answer"]
                sources = {(doc.metadata["source"], doc.metadata["page"]) 
                          for doc in response["context"]}

                # Formatted Response
                st.markdown(
                    f'<div class="chat-message bot-message">{answer}</div>',
                    unsafe_allow_html=True
                )

                # Performance
                st.markdown(
                    f'<div class="performance-metric">'
                    f'Generated in {time.time() - start_time:.2f}s '
                    f'using {len(sources)} sources'
                    f'</div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()