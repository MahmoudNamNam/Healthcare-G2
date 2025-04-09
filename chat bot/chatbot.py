import sys
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 🎯 Initialize Ollama Model for Psychology Q&A
llm = OllamaLLM(model="Koesn/llama3-8b-instruct")  # Change to your Ollama model

# 🧠 Psychology Q&A Prompt Template
prompt = ChatPromptTemplate.from_template(
    """You are a friendly and empathetic AI therapist. 
    Your goal is to provide short, human-like, and supportive responses.
    Keep replies warm, understanding, and natural, like a real therapist. 

    User: {user_input}
    Assistant:"""
)

# 🔄 Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# 🗣️ Create the LangChain Conversation
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Terminal Chatbot Loop
print("\n🧠 Welcome to the Psychology Chatbot!")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Assistant: Take care! If you ever need support, I'm here. 😊")
        break
    
    # 🤖 Generate psychology response
    response = conversation.invoke({"user_input": user_input})

    # Display response
    print(f"Assistant: {response['text']}\n")
