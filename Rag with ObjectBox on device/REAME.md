# DocuMind AI - Intelligent Document Analysis

DocuMind AI is a powerful document analysis tool that allows you to upload multiple PDFs and ask questions about their content. It uses advanced AI to provide accurate, context-aware answers while maintaining cross-document awareness to reduce hallucinations.

## Features

- **Multi-Document Analysis**: Upload and analyze multiple PDFs simultaneously.
- **Cross-Document Context**: AI understands relationships between documents for better accuracy.
- **Source References**: Every answer includes references to the source document and page number.
- **Fast Processing**: Built with Groq's Mixtral-8x7b model for quick responses.
- **Minimalist UI**: Clean and professional interface for seamless user experience.

## How It Works

1. **Document Upload**:
   - Upload one or more PDF documents through the sidebar.
   - Click "Process Documents" to analyze and prepare them for querying.

2. **AI-Powered Querying**:
   - Ask questions about the uploaded documents in natural language.
   - The system retrieves relevant information from all documents and provides a concise answer.

3. **Cross-Document Awareness**:
   - The AI analyzes relationships between documents to provide context-aware answers.
   - It identifies contradictions, supporting evidence, and temporal sequences across documents.

4. **Source Attribution**:
   - Every response includes references to the source document and page number.
   - Expand the "Document References" section to see all sources used.

5. **Performance Metrics**:
   - Response time and number of sources used are displayed for transparency.

## Tech Stack

- **AI Model**: Mixtral-8x7b via Groq API
- **Vector Database**: ObjectBox for document embeddings
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2`
- **Frontend**: Streamlit for the web interface
- **Backend**: Python with LangChain for document processing and retrieval

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/documind-ai.git
   cd documind-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your Groq API key:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

## Usage

1. Upload PDFs through the sidebar.
2. Click "Process Documents" to analyze them.
3. Ask questions in the chat input box.
4. View answers with source references and performance metrics.

---

This `README.md` provides a clear overview of the project, its features, and how to set it up. Let me know if you'd like to add anything else!