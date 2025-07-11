import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Set OpenAI API key
load_dotenv()
# Initialize Streamlit app
st.title("Dr. Ali AI Assistant")
st.write("Hello! I'm Dr. Ali's AI Assistant. Ask me about Dr. Mohammad Azhar Ali, his services, or anything related to Amae Plastic Surgery Center and AMAE Med Spa!")

# Initialize session state for chat history and conversation ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Define greeting responses
GREETING_RESPONSES = {
    "hello": "Hello! How can I assist you today regarding Dr. Ali's services?",
    "hi": "Hi there! Feel free to ask about Dr. Ali's medical services or any related questions.",
    "how are you": "I'm doing great, thanks for asking! How about you? Ready to explore Dr. Ali's services?",
    "bye": "Goodbye! Feel free to come back if you have more questions about Dr. Ali's services.",
    "goodbye": "Take care! I'm here whenever you need help with Dr. Ali's information."
}

# Function to check if input is a greeting
def is_greeting(user_input):
    user_input = user_input.lower().strip()
    return user_input in GREETING_RESPONSES

# Function to load and process the document
@st.cache_resource
def load_documents():
    loader = TextLoader("dr_ali_text.txt" , encoding="utf-8")  # Ensure this file is in the same directory
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Initialize the vectorstore
vectorstore = load_documents()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Initialize conversation memory with explicit message storage
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Initialize the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True
)

# Load previous chat history from file if it exists
def load_chat_history():
    history_file = f"chat_history_{st.session_state.conversation_id}.txt"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("user:"):
                    content = line.replace("user: ", "").strip()
                    memory.chat_memory.add_user_message(content)
                    st.session_state.messages.append({"role": "user", "content": content})
                elif line.startswith("assistant:"):
                    content = line.replace("assistant: ", "").strip()
                    memory.chat_memory.add_ai_message(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})

# Load chat history at startup
load_chat_history()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("Type your question or message here...")

if user_input:
    # Add user message to session state and memory
    st.session_state.messages.append({"role": "user", "content": user_input})
    memory.chat_memory.add_user_message(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Check if input is a greeting
    if is_greeting(user_input):
        response = GREETING_RESPONSES[user_input.lower().strip()]
    else:
        # Use RAG to get document-based response with full chat history
        result = qa_chain({"question": user_input})
        response = result["answer"]

    # Add assistant response to session state and memory
    st.session_state.messages.append({"role": "assistant", "content": response})
    memory.chat_memory.add_ai_message(response)
    with st.chat_message("assistant"):
        st.markdown(response)

# Save chat history to a file
def save_chat_history():
    with open(f"chat_history_{st.session_state.conversation_id}.txt", "w") as f:
        for message in st.session_state.messages:
            f.write(f"{message['role']}: {message['content']}\n")

if st.session_state.messages:
    save_chat_history()