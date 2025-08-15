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
    "Upload your documents (PDF, DOCX, TXT)",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt"]
)

# Process uploaded files and store in session_state
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in [doc.get("name") for doc in st.session_state['documents']]:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(uploaded_file)
                pages = loader.load()
                for page in pages:
                    page.metadata["name"] = uploaded_file.name
                st.session_state['documents'].extend(pages)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(uploaded_file)
                st.session_state['documents'].append({"page_content": text, "metadata": {"name": uploaded_file.name}})
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
                st.session_state['documents'].append({"page_content": text, "metadata": {"name": uploaded_file.name}})

# Only proceed if documents exist
if st.session_state['documents']:
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
    docs = text_splitter.split_documents(st.session_state['documents'])

    # Create vector DB
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory='./data'
    )
    vectordb.persist()

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
