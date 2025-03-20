# rag_app.py
import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
api_key_pine = os.getenv("PINE_API_KEY")

# ---------------------------------------------------------
# NLTK Setup
# ---------------------------------------------------------
import nltk
nltk_dir = "C:/Users/Armghan/nltk_data"
os.makedirs(nltk_dir, exist_ok=True)
nltk.download('stopwords', download_dir=nltk_dir)
nltk.download('punkt', download_dir=nltk_dir)
nltk.data.path.append(nltk_dir)
os.environ['NLTK_DATA'] = nltk_dir

# ---------------------------------------------------------
# Pinecone, Embeddings, and BM25 Encoder Setup
# ---------------------------------------------------------
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

pc = Pinecone(api_key=api_key_pine)
INDEX_NAME = "langchain-hybrid-search-pinecone"

if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BM25 Encoder in session state
if 'bm25encoder' not in st.session_state:
    if os.path.exists("bm25encoder_values.json"):
        st.session_state.bm25encoder = BM25Encoder().load("bm25encoder_values.json")
    else:
        st.session_state.bm25encoder = BM25Encoder()

# ---------------------------------------------------------
# Document Processing Functions
# ---------------------------------------------------------
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and return chunks"""
    all_texts = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                all_texts.extend([doc.page_content for doc in split_docs])
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
    return all_texts

# ---------------------------------------------------------
# Groq LLM Setup
# ---------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768",
    temperature=0.5
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an **expert AI assistant** with deep knowledge in analyzing and interpreting documents. Your task is to provide **accurate, insightful, and professional answers** based on the given context. Follow these rules:

        1. **Be Accurate**:
           - Only use information from the provided context. Do not make up facts or provide answers outside the scope of the context.
           - If the context does not contain enough information, say: "I don't know" or "The context does not provide enough information."

        2. **Adapt Your Response Style**:
           - For **fact-based questions** (e.g., "What is X?"), provide a **direct and concise answer**.
           - For **analytical or complex questions** (e.g., "Why did X happen?" or "What are the implications of X?"), provide a **detailed explanation** with meaningful insights, reasoning, and implications.
           - For **comparison questions** (e.g., "Compare X and Y"), highlight key differences and similarities in a structured manner.

        3. **Be Professional**:
           - Use **formal language** and avoid informal expressions.
           - Structure your response logically, using **bullet points**, **numbered lists**, or **tables** if appropriate.
           - Avoid unnecessary jargon unless it is relevant to the context.

        4. **Provide Insights**:
           - If the question requires deeper analysis, explain the **causes**, **effects**, or **implications** behind the answer.
           - Highlight **trends**, **patterns**, or **anomalies** in the data if applicable.
           - Offer **actionable insights** or **recommendations** where relevant.

        5. **Source Attribution**:
           - At the end of your answer, include a **creative and professional note** about the source of the information. For example:
             - "Source: Extracted from the 2022 American Community Survey (ACS) report."
             - "Based on data from the U.S. Census Bureau's ACS brief."
             - "Information derived from the provided document on health insurance coverage."

        6. **Handle Ambiguity Gracefully**:
           - If the question is ambiguous or unclear, ask for clarification or provide a **general answer** based on the most likely interpretation.
           - If the context contains conflicting information, acknowledge the discrepancy and provide a balanced view.

        7. **Be Concise but Thorough**:
           - Avoid unnecessary verbosity, but ensure the answer is complete and covers all aspects of the question.
           - Use **headings** or **subheadings** to organize complex answers.

        8. **Examples of Expert Responses**:
           - For a question like "What are the top three occupation groups?", respond with:
             - "The top three occupation groups are:
               1. Management (11.3%)
               2. Office and administrative support (10.2%)
               3. Sales and related (9.5%)
               Source: 2018 Survey of Income and Program Participation (SIPP)."

           - For a question like "Why do workers in food preparation and serving related occupations have lower earnings?", respond with:
             - "Workers in food preparation and serving related occupations tend to have lower earnings due to several factors:
               - **High rates of part-time work**: Nearly 50% of workers in this group work part-time, which reduces their overall earnings.
               - **Reliance on tips**: Many workers in this group rely on tips, which can be unpredictable and vary based on customer behavior.
               - **Nonstandard schedules**: A significant portion of workers in this group have nonstandard or unpredictable work schedules, which may limit their ability to work additional hours or secure higher-paying jobs.
               Source: 2018 SIPP data."

        9. **Creative and Professional Tone**:
           - Use a tone that is **authoritative** yet **approachable**.
           - Avoid overly technical language unless the question demands it.
           - End your response with a **polish** that leaves the user feeling informed and satisfied.
    """),
    ("human", """
        **Context**: {context}

        **Question**: {query}

        **Instructions**:
        - Analyze the context carefully.
        - Provide a clear, concise, and professional answer to the question.
        - If the question requires deeper analysis, provide meaningful insights and explanations.
        - If the question cannot be answered from the context, say "I don't know" or "The context does not provide enough information."
        - Include a creative and professional source attribution at the end of your answer.
    """)
])
# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    .stTextArea textarea {border-radius: 10px;}
    .success {color: #28a745;}
    .error {color: #dc3545;}
    .answer-box {background-color: #transparent; padding: 20px; border-radius: 10px; color: #ffffff;} /* Updated this line */
    </style>
    """, unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF files for processing"
    )
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Process uploaded files
                new_texts = process_uploaded_files(uploaded_files)
                
                if new_texts:
                    # Update BM25 encoder with all texts
                    if 'all_texts' not in st.session_state:
                        st.session_state.all_texts = []
                    st.session_state.all_texts.extend(new_texts)
                    
                    # Fit and save BM25 encoder
                    st.session_state.bm25encoder.fit(st.session_state.all_texts)
                    st.session_state.bm25encoder.dump("bm25encoder_values.json")
                    
                    # Initialize retriever with updated encoder
                    retriever = PineconeHybridSearchRetriever(
                        embeddings=embeddings,
                        sparse_encoder=st.session_state.bm25encoder,
                        index=index
                    )
                    
                    # Index new documents
                    try:
                        retriever.add_texts(new_texts)
                        st.session_state.retriever = retriever
                        st.success(f"✅ Processed {len(new_texts)} text chunks!")
                        st.session_state.processed = True
                    except Exception as e:
                        st.error(f"Indexing error: {str(e)}")
                else:
                    st.warning("No valid text extracted from documents")
        else:
            st.warning("Please upload PDF files first")

# Main interface
st.title("🧠 Enterprise DocuMind AI")
st.markdown("""
    **Powered by Groq's Mixtral 8x7B AI**  
    Advanced document understanding with hybrid semantic search
    """)

# Initialize retriever in session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=st.session_state.bm25encoder,
        index=index
    )

# Query input and response
query = st.text_input(
    "📝 Enter your question:",
    placeholder="Ask anything about your documents...",
    key="query_input"
)

if st.button("🔍 Search Knowledge Base"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        # Set up RAG chain
        rag_chain = (
            {"context": st.session_state.retriever, "query": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        
        with st.spinner("Analyzing documents..."):
            try:
                response = rag_chain.invoke(query)
                st.subheader("📄 Answer")
                st.markdown(f"<div class='answer-box'>{response.content}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Display document stats
if 'all_texts' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"📑 **Documents Processed:** {len(st.session_state.all_texts)} chunks")
    st.sidebar.markdown(f"📊 **Index Status:** {INDEX_NAME}")