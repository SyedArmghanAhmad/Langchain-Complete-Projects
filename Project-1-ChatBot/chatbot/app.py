
# Importing necessary libraries from langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import streamlit as st

# Load .env file
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")


#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please Answer the Queries to the best of your abilities."),
        ("user","Question :{question}"),
    ]
)

# Streamlit app
# Streamlit UI Customization
st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: #0d1117;
            color: #e6edf3;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #0d1117;
            color: #e6edf3;
        }
        .stTextInput>div>div>input {
            background-color: #161b22;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #30363d;
        }
        .stButton>button {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            border-radius: 8px;
            padding: 8px;
            border: none;
        }
        .stTitle {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .message-box {
            border-radius: 10px;
            padding: 12px;
            margin: 10px 0;
            font-size: 16px;
        }
        .user-message {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            text-align: right;
            padding: 12px;
            border-radius: 12px;
            max-width: 75%;
            display: inline-block;
        }
        .bot-message {
        background: linear-gradient(135deg, #374151, #4b5563); /* Dark Gray to Muted Blue */
        color: white;
        text-align: left;
        padding: 12px;
        border-radius: 12px;
        max-width: 75%;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0px 3px 8px rgba(75, 85, 99, 0.3); /* Softer shadow */
}

}
    </style>
    """,
    unsafe_allow_html=True
)



# Chat header
st.markdown("<h1 class='stTitle'>🤖 AI ChatBot (Mixtral LLM)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Powered by LangChain & Groq 🚀</p>", unsafe_allow_html=True)

#Groq LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768")

output_parser = StrOutputParser()
chain=prompt|llm|output_parser

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role, content = msg
    if role == "user":
        st.markdown(f"<div class='message-box user-message'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='message-box bot-message'>{content}</div>", unsafe_allow_html=True)

# User input
input_text = st.text_input("Type your message...", key="input_text")
if input_text:
    # Display user message
    st.session_state.messages.append(("user", input_text))
    st.markdown(f"<div class='message-box user-message'>{input_text}</div>", unsafe_allow_html=True)

    # Get response from the chatbot
    response = chain.invoke({'question': input_text})

    # Display bot response
    st.session_state.messages.append(("bot", response))
    st.markdown(f"<div class='message-box bot-message'>{response}</div>", unsafe_allow_html=True)

# Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.pop("messages", None)  # Completely remove chat history
    st.session_state.pop("input_text", None)  # Clear input field if stored
    st.rerun()  # Refresh the app



