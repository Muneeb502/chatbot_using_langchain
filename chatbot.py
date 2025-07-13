## chatbot.py

import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.document_loaders import TextLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Read OpenAI API key securely (Streamlit Cloud Secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit Secrets.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm Dr. Ali's AI Assistant. Ask me about Dr. Mohammad Azhar Ali, his services, or anything related to Amae Plastic Surgery Center and AMAE Med Spa!"}
    ]

# Load and process document + build vector store (FAISS)
# @st.cache_resource
# def load_vector_store():
#     try:
#         loader = TextLoader("dr_ali_text.txt", encoding="utf-8")
#         docs = loader.load()
#     except FileNotFoundError:
#         st.error("Document file 'dr_ali_text.txt' not found. Please ensure it exists in the correct directory.")
#         st.stop()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
#     chunks = text_splitter.split_documents(docs)

#     vector_store = FAISS.from_documents(
#         documents=chunks,
#         embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
#     )
#     return vector_store

# # Initialize retriever and model
# vector_store = load_vector_store()
@st.cache_resource
def load_documents():
    # Use PyPDFLoader instead of TextLoader
    loader = PyPDFLoader("dr_ali_text.pdf")  # Make sure this is a PDF file now
    pages = loader.load_and_split()  # This automatically splits by pages
    
    # Each page becomes one document/chunk
    # No need for RecursiveCharacterTextSplitter since we're keeping pages intact
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(pages, embeddings)
    return vectorstore

# Initialize the vectorstore
vectorstore = load_documents()


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
model = ChatOpenAI(temperature=0.3, api_key=openai_api_key)

# Updated Prompt Template
prompt = PromptTemplate(
    template="""**Role**: You are an AI assistant for a Retrieval-Augmented Generation (RAG) system, assisting users with information about Dr. Mohammad Azhar Ali, Amae Plastic Surgery Center, and AMAE Med Spa. Your responses must be **accurate, context-aware, concise, and polite**.

### **Instructions**:
1. **Source of Truth**:
   - **Strictly use only the provided `{data}`** to answer questions about Dr. Ali, his services, or the practice.
   - If the answer isn't in the data, respond:  
     ❌ "I'm sorry, I don't have that information."
   - For follow-up questions, use the chat history to maintain context.

2. **Allowed Free Responses**:
   - Greetings, farewells, and general small talk are fine without relying on data.

3. **Response Style**:
   - Be professional, helpful, and concise.
   - Ensure responses are user-friendly and clear.

### **Chat History (Last 4 messages) reply according to the chat history and from given data if user are asking about anything related to previous messages you know you have to continue the conversation**:
{chat_history}

### **User's Latest Question**:
{query}

### **Retrieved Data**:
{data}

### **Assistant's Response**:""",
    input_variables=["data", "query", "chat_history"]
)

# Format retrieved documents
def doc_format(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

# Format chat history using AIMessage and HumanMessage
def get_chat_history(_):
    messages = st.session_state.get("messages", [])
    # Take the last 4 messages or fewer if not enough
    last_msgs = messages[-4:] if len(messages) > 0 else []
    history = []
    for msg in last_msgs:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    # Convert messages to a string representation for the prompt
    history_str = ""
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    return history_str.strip()

# LangChain Pipeline
parallel_chain = RunnableParallel({
    "data": retriever | RunnableLambda(doc_format),
    "query": RunnablePassthrough(),
    "chat_history": RunnableLambda(get_chat_history)
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser

# Streamlit UI
# st.title("Dr. Ali AI Assistant")
# st.markdown("Welcome to the official chatbot for Amae Plastic Surgery Center and AMAE Med Spa. Ask about Dr. Mohammad Azhar Ali, our services")
st.markdown(
    """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;">
        <h1 style="text-align: center;  text-size: 30px">Dr. Ali AI Assistant</h1>
        <p style="text-align: center; font-size: 1.2em;">
            Welcome to the official chatbot for Amae Plastic Surgery Center and AMAE Med Spa.<br>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input and response
if user_input := st.chat_input("Ask about Dr. Ali, services, or anything else..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = main_chain.invoke(user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                response = "I'm sorry, something went wrong. Please try again."
        
        st.session_state.messages.append({"role": "assistant", "content": response})


