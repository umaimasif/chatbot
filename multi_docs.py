import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message  # pip install streamlit_chat
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import asyncio

# Setup asyncio event loop for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# ==== LLM and embeddings ====
llm_model = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(
    temperature=0.0,
    model=llm_model,
    google_api_key=gemini_api_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# ==== Streamlit front-end ====
st.title("Docs QA Bot")
st.header("Ask anything about your documents... 🤖")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDFs, Word, or Text files",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    new_docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
            new_docs.extend(loader.load())
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(uploaded_file)
            new_docs.append(Document(page_content=text))
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
            new_docs.append(Document(page_content=text))
    
    # Split only the new documents
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
    new_chunks = text_splitter.split_documents(new_docs)
    
    # Append new chunks to existing session memory
    st.session_state['documents'].extend(new_chunks)
    
    st.success(f"Loaded {len(new_chunks)} new chunks. Total in memory: {len(st.session_state['documents'])}")

# Only proceed if there are documents in memory
if st.session_state['documents']:
    # Create vector DB from all chunks in memory
    vectordb = Chroma.from_documents(
        documents=st.session_state['documents'],
        embedding=embedding,
    )
    

    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    # Get user query
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        result = qa_chain({'question': user_input, 'chat_history': st.session_state['chat_history']})
        st.session_state['chat_history'].append((user_input, result['answer']))
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result['answer'])

# Display chat
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
