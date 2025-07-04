import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Read OpenAI API key securely (Streamlit Cloud Secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm DR_ALI_AI Assistant. How can I help you today?"}]

# Load and process document + build vector store (FAISS)
@st.cache_resource
def load_vector_store():
    loader = TextLoader("temp_2.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )

    return vector_store

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
model = ChatOpenAI(temperature=0.3, api_key=openai_api_key)

# Prompt Template
prompt = PromptTemplate(
    template="""**Role**: You are an AI assistant for a Retrieval-Augmented Generation (RAG) system. Your responses must be **accurate, context-aware, and concise**.

### **Instructions**:
1. **Source of Truth**:
   - **Strictly use only the provided `{data}`** to answer questions.
   - **Never hallucinate**—if the answer isn't in the data, say:  
     ❌ "I don't know."

2. **Allowed Free Responses (No Data Needed)**:
   - Greetings, farewells, and small talk are fine.

3. **Response Style**:
   - Be helpful but precise.
   - If unsure: say "I don't know."

### **Chat History**:
{chat_history}

### **User's Latest Question**:
{query}

### **Assistant's Response**:""",
    input_variables=["data", "query", "chat_history"]
)

# Format retrieved docs
def doc_format(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

# Last few lines of chat history
def get_chat_history(_):
    messages = st.session_state.get("messages", [])
    last_msgs = messages[-4:] if len(messages) >= 1 else []
    history = ""
    for msg in last_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history.strip()

# LangChain Pipeline
parallel_chain = RunnableParallel({
    "data": retriever | RunnableLambda(doc_format),
    "query": RunnablePassthrough(),
    "chat_history": RunnableLambda(get_chat_history)
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser

# Streamlit UI
st.title("DR_ALI_AI Assistant")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input and response
if user_input := st.chat_input("Message DR_ALI_AI..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = main_chain.invoke(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

