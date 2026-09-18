import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY_PROMPT")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.5,
    max_tokens=5000
)

st.title("🤖 Chat with Virat Kewal")

# Initialize memory
if "messages" not in st.session_state:
    st.session_state.messages = []

system_message = SystemMessage(
    content="""
    Your name is Virat Kewal and you are boyfreind of Diya Upreti.
    If the user asks:
    - What is your name?
    - Who are you?
    - What should I call you?
    - Tell me your name.

    Always answer that your name is Virat Kewal.
    """
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:

    # Add user message to memory
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # System prompt + complete conversation history
    messages = [system_message] + st.session_state.messages

    # Invoke model
    result = model.invoke(messages)

    # Get AI response
    ai_response = result.content

    # Add AI response to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

    # Display AI response
    with st.chat_message("assistant"):
        st.write(ai_response)
