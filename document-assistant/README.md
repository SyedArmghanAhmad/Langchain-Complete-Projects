
# 📚 Context-Aware Document Assistant

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FF6F61?style=for-the-badge&logo=LangChain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-00A98F?style=for-the-badge&logo=FAISS&logoColor=white)

**Context-Aware Document Assistant** is a powerful Retrieval-Augmented Generation (RAG) application that allows users to interact with documents (PDFs or websites) using natural language queries. Built with **Streamlit**, **LangChain**, and **Hugging Face**, this app leverages advanced AI models to provide accurate, context-aware responses.

---

## 🌟 Features

- **Hybrid Document Contexts**: Seamlessly switch between PDFs and website content.
- **Smart Document Processing**: Automatically processes and indexes uploaded PDFs or website URLs.
- **Advanced AI Models**: Utilizes state-of-the-art models like **Mixtral-8x7b** and **Gemma2-9b** for accurate responses.
- **Metadata-Aware Responses**: Displays sources and page numbers for every answer.
- **User-Friendly Interface**: Intuitive Streamlit UI with real-time chat and document management.
- **Customizable Settings**: Adjust chunk size, overlap, and AI model to suit your needs.

---

## 🚀 Quick Start

### Prerequisites

Before running the app, ensure you have the following installed:

- Python 3.8+
- Docker (optional, for containerized deployment)
- A **Groq API Key** (for using Mixtral/Gemma models)

### Installation

1. **Clone the Project**:

   ```bash
   git clone https://github.com/SyedArmghanAhmad/Langchain-Complete-Projects.git
   cd Langchain-Complete-Projects/document-assistant
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the `frontend` directory.
   - Add your Groq API key:

     ```plaintext
     GROQ_API_KEY=your_groq_api_key_here
     ```

3. **Install Dependencies**:
   - For the backend:

     ```bash
     cd backend
     pip install -r requirements.txt
     ```

   - For the frontend:

     ```bash
     cd ../frontend
     pip install -r requirements.txt
     ```

4. **Run the App**:
   - Start the backend:

     ```bash
     cd ../backend
     python main.py
     ```

   - Start the frontend:

     ```bash
     cd ../frontend
     streamlit run App.py
     ```

5. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

---

## 🛠️ How It Works

### 1. **Document Upload & Processing**

- Upload multiple PDFs or provide a website URL.
- The app processes the documents using **Hugging Face embeddings** and indexes them using **FAISS** for efficient retrieval.

### 2. **Natural Language Queries**

- Ask questions about the uploaded documents or website content.
- The app uses **LangChain** and **Groq AI models** to generate accurate, context-aware responses.

### 3. **Source Attribution**

- Every response includes the source document and page number(s) for transparency and verification.

### 4. **Customizable Settings**

- Adjust chunk size, overlap, and AI model to optimize performance for your use case.

---

## 🖥️ User Interface

### **Sidebar**

- **Context Selection**: Choose between PDFs or website content.
- **Document Upload**: Upload multiple PDFs for processing.
- **Settings**: Adjust chunk size, overlap, and AI model.

### **Main Chat Interface**

- **Real-Time Chat**: Interact with the AI in a conversational manner.
- **Source Display**: View the source document and page numbers for every response.

---

## 🧠 Supported AI Models

- **Mixtral-8x7b-32768**: A high-performance, open-source LLM for advanced use cases.
- **Gemma2-9b-it**: A lightweight yet powerful model for faster responses.

---

## 📂 Project Structure

```plaintext
document-assistant/
├── backend/                  # Backend logic and document processing
│   ├── main.py               # Main backend script
│   ├── requirements.txt      # Backend dependencies
│   ├── cache/                # Cached data (if any)
│   └── faiss_index/          # FAISS index storage
│       ├── index.faiss       # FAISS index file
│       └── index.pkl         # FAISS metadata
├── frontend/                 # Frontend Streamlit app
│   ├── App.py                # Main Streamlit application
│   ├── requirements.txt      # Frontend dependencies
│   ├── .env                  # Environment variables
│   └── Dockerfile            # Docker configuration for frontend
```

---

## 🐳 Docker Deployment

To deploy the app using Docker:

1. **Build the Docker Image**:

   ```bash
   cd frontend
   docker build -t document-assistant .
   ```

2. **Run the Docker Container**:

   ```bash
   docker run -p 8501:8501 document-assistant
   ```

3. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Streamlit** for the amazing framework.
- **LangChain** for simplifying AI workflows.
- **Hugging Face** for open-source NLP tools.
- **Groq** for providing access to cutting-edge AI models.

---

## 📧 Contact

For questions or feedback, feel free to reach out:

- **Email**: <syedarmghanahmad.work@gmail.com>
- **GitHub**: [SyedArmghanAhmad](https://github.com/SyedArmghanAhmad)

---

Enjoy using the **Context-Aware Document Assistant**! 🚀

---
