import requests
import streamlit as st

def get_Groq_response(input_text):
    response = requests.post("http://localhost:8000/groq/invoke",
    json={'input': input_text})

    return response.json()['output']['content']

# Set Page Config
st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="centered")

# Custom Styling with Animations
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #0d1117;
            color: #e6edf3;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            background-color: #161b22;
            color: white;
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #30363d;
            font-size: 16px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            border-radius: 8px;
            padding: 10px;
            border: none;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #a777e3, #6e8efb);
        }
        .stTitle {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .message-container {
        display: flex;
        flex-direction: column;
        gap: 20px; /* Increased gap for proper spacing */
        padding: 10px;
        margin-top: 20px;
    }
        .message-box {
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        max-width: 75%;
        word-wrap: break-word;
        opacity: 0;
        transform: translateY(10px);
        animation: fadeInUp 0.4s ease-out forwards;
        margin-bottom: 12px; /* Ensures space even after animation */
}
        .user-message {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            text-align: right;
            align-self: flex-end;
            margin-left: auto;
            box-shadow: 0px 3px 6px rgba(174, 139, 250, 0.3);
        }
        .bot-message {
            background: linear-gradient(135deg, #374151, #4b5563);
            color: white;
            text-align: left;
            align-self: flex-start;
            margin-right: auto;
            font-weight: 500;
            box-shadow: 0px 3px 8px rgba(75, 85, 99, 0.3);
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(5px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Chat header
st.markdown("<h1 class='stTitle'>🤖 AI ChatBot (Mixtral LLM)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Powered by LangChain & Groq 🚀</p>", unsafe_allow_html=True)


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
    response = get_Groq_response(input_text)

    # Display bot response
    st.session_state.messages.append(("bot", response))
    st.markdown(f"<div class='message-box bot-message'>{response}</div>", unsafe_allow_html=True)

# Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.pop("messages", None)  # Completely remove chat history
    st.session_state.pop("input_text", None)  # Clear input field if stored
    st.rerun()  # Refresh the app

