import streamlit as st
import asyncio
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma:7b")
# MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))


# ------------------------------
# Section: Conversational Chatbot
# ------------------------------

# Initialize Ollama LLM
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=TEMPERATURE,
    num_ctx=8192
)

# Define prompt template with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Provide concise, accurate, and friendly responses."),
    ("human", "{history}\n\nCurrent message: {input}")
])

# Async function to stream chatbot response
async def stream_chatbot(input_text: str, placeholder, history: list):
    if not input_text.strip():
        placeholder.markdown("Error: Please enter a valid query.")
        return

    try:
        # Format chat history for the prompt
        history_text = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        # Initialize response
        response = ""
        placeholder.markdown(response)
        
        # Stream response chunks
        async for chunk in llm.astream(prompt.format_messages(history=history_text, input=input_text)):
            content = chunk.content
            response += content
            placeholder.markdown(response)
        return response[10:]  # Return response without "Assistant: " prefix

    except Exception as e:
        placeholder.markdown(f"Error: {str(e)}")
        return None

# ------------------------------
# Section: Streamlit App
# ------------------------------

st.title("Ollama Chatbot")
st.write("Chat with the Gemma:7b model powered by Ollama. Type your message below.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Run async streaming with chat history
        response = asyncio.run(stream_chatbot(user_input, placeholder, st.session_state.messages))
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})