import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

#  Load Ollama model
llm = OllamaLLM(model="Koesn/llama3-8b-instruct") 

# Enhanced Therapist Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
You are a compassionate, thoughtful, and supportive AI therapist.
Your role is to make people feel heard, understood, and comforted. 
Use gentle, kind language — keep responses short and empathetic, as if you're in a real therapy session.

Always speak in the tone of a calm, professional therapist — warm, encouraging, and never robotic.

Here is the ongoing conversation:
{chat_history}

User: {user_input}
Therapist:"""
)

#  Memory to keep chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#  LLMChain for the therapy conversation
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# Streamlit Frontend
st.set_page_config(page_title="EmpathyBot - AI Therapist", page_icon="🧠")
st.title("🧠 EmpathyBot: Your AI Therapist")
st.markdown("Welcome. I'm here to listen and support you. Feel free to share what's on your mind. 💬")

#  Session state to store full conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    response = conversation.invoke({"user_input": user_input})
    therapist_reply = response["text"].strip()

    # Save and show assistant message
    st.session_state.messages.append({"role": "assistant", "content": therapist_reply})
    with st.chat_message("assistant"):
        st.markdown(therapist_reply)
