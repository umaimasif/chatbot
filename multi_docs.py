import os
from dotenv import find_dotenv, load_dotenv
import docx2txt
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from streamlit_chat import message
import asyncio
import numpy as np

# Setup asyncio loop for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and embeddings
llm_model = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(
    temperature=0.0,
    model=llm_model,
    google_api_key=gemini_api_key
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# Streamlit setup
st.title("Docs QA Bot")
st.header("Upload your documents and ask questions 🤖")
import os
import json
from langchain.schema import Document

CACHE_FILE = "uploaded_docs.json"

def save_uploaded_docs(docs):
    # Convert Document objects to dicts
    doc_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
def load_cached_docs():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            doc_dicts = json.load(f)
        return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in doc_dicts]
    return []

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = []  # List of Document objects
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []  # List of embedding vectors
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDFs, DOCX, or TXT files", 
    type=['pdf', 'docx', 'txt'], 
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
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

    # Split documents into chunks
    chunked_docs = text_splitter.split_documents(new_docs)

    # Add to session memory
    st.session_state['documents'].extend(chunked_docs)

    # Generate embeddings for new chunks and store
    new_embeddings = embedding_model.embed_documents([doc.page_content for doc in chunked_docs])
    st.session_state['embeddings'].extend(new_embeddings)

    st.success(f"Loaded {len(chunked_docs)} new chunks. Total in memory: {len(st.session_state['documents'])}")

# Function to retrieve most relevant docs from memory
def retrieve_relevant_docs(query, top_k=5):
    if not st.session_state['documents']:
        return []
    query_embedding = embedding_model.embed_query(query)
    # Compute cosine similarity
    similarities = np.array([np.dot(query_embedding, emb)/(np.linalg.norm(query_embedding)*np.linalg.norm(emb))
                             for emb in st.session_state['embeddings']])
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [st.session_state['documents'][i] for i in top_indices]

# Get user query
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    relevant_docs = retrieve_relevant_docs(user_input)
    if relevant_docs:
        # Run LLM on top relevant docs
        class InMemoryRetriever:
             def __init__(self, docs):
                self.docs = docs

             def get_relevant_documents(self, query):
               return self.docs

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=InMemoryRetriever(relevant_docs), # Pass the docs directly
            return_source_documents=True,
            verbose=False
        )
        result = qa_chain({'question': user_input, 'chat_history': st.session_state['chat_history']})
        st.session_state['chat_history'].append((user_input, result['answer']))
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result['answer'])
    else:
        st.warning("No documents uploaded yet!")

# Display chat
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
