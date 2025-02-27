import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

import streamlit as st
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import torch

# Cache the model and other heavy objects
@st.cache_resource
def load_model_and_components():
    # Step 1: Load Llama model and tokenizer from Hugging Face
    model_name = "/datadisk/evaluation/models/Llama-2-7b-chat-hf/"  # Replace with your Llama model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Step 2: Create a Hugging Face pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7
    )

    # Wrap the pipeline in a LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Step 3: Load documents from the text file
    file_path = "data/banking_docs.txt"
    loader = TextLoader(file_path)
    documents = loader.load()

    # Step 4: Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    # Step 5: Generate embeddings using a dedicated embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 6: Create the FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Step 7: Set up Prompt Management
    prompt_template = """You are a helpful banking assistant. Use the following information to answer the user's question in a concise and helpful way. If you don't know the answer, say "I don't know."

    Relevant Information:
    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template
    )

    # Step 8: Set up Memory Management
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Step 9: Set up the RetrievalQA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Retrieve only top 2 documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Load the model and components once
qa_chain = load_model_and_components()

# Step 10: Streamlit Chat Interface
#st.image("logo.png", width=100)  # Replace with your logo path
st.title("Banking Chatbot 🏦")
st.write("Welcome to the Banking Chatbot! Ask me anything about banking.")

# Initialize chat session management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}  # Initialize with a default chat session
    st.session_state.current_session = "Chat 1"

# Add a sidebar for chat session management
with st.sidebar:
    # Header Section: New Chat Button
    st.header("Chat Sessions")
    if st.button("➕ New Chat"):
        new_session_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_session_id] = []
        st.session_state.current_session = new_session_id
    
    # Middle Section: Scrolling Chat List
    st.subheader("Your Chats")
    chat_container = st.container()
    with chat_container:
        for session_id in st.session_state.chat_sessions:
            if st.button(session_id, key=session_id):
                st.session_state.current_session = session_id

    # Footer Section: About and Clear Chat Button
    st.subheader("Settings")
    if st.button("🧹 Clear Current Chat"):
        st.session_state.chat_sessions[st.session_state.current_session] = []

    st.subheader("About")
    st.write("This is a banking chatbot powered by Llama 2 and LangChain. It can answer questions about banking services, accounts, loans, and more.")

# Initialize chat history for the current session
if st.session_state.current_session not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_session] = []

# Display a welcome message if the chat history is empty
if not st.session_state.chat_sessions[st.session_state.current_session]:
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": "Hello! I'm your banking assistant. How can I help you today?"})

# Display chat messages from history
for message in st.session_state.chat_sessions[st.session_state.current_session]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].split("Answer:")[-1].strip())

# Simulate typing animation
def simulate_typing(text):
    placeholder = st.empty()
    for i in range(len(text) + 1):
        placeholder.markdown(text[:i])
        time.sleep(0.02)  # Adjust the speed of the typing animation

# Accept user input
if user_query := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate a response
    response = qa_chain.invoke(user_query)["result"]

    # Add assistant response to chat history
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response})
    # Display assistant response with typing animation
    with st.chat_message("assistant"):
        simulate_typing(response.split("Answer:")[-1].strip())

# Add feedback buttons
if st.session_state.chat_sessions[st.session_state.current_session] and st.session_state.chat_sessions[st.session_state.current_session][-1]["role"] == "assistant":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Thumbs Up"):
            st.write("Thank you for your feedback!")
    with col2:
        if st.button("👎 Thumbs Down"):
            st.write("We're sorry to hear that. Please let us know how we can improve.")

# Add a download button for chat history
if st.session_state.chat_sessions[st.session_state.current_session]:
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_sessions[st.session_state.current_session]])
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name=f"{st.session_state.current_session}_history.txt",
        mime="text/plain"
    )