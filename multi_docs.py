import os
from dotenv import find_dotenv, load_dotenv
import docx2txt
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document, BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from streamlit_chat import message
import asyncio
import numpy as np
import json

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

CACHE_FILE = "uploaded_docs.json"

def save_uploaded_docs(docs):
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
    st.session_state['documents'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []
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

# Custom in-memory retriever
class InMemoryRetriever(BaseRetriever):
    _docs: list = []
    _embeddings: list = []
    _top_k: int = 5

    def set_memory(self, docs, embeddings, top_k=5):
        self._docs = docs
        self._embeddings = embeddings
        self._top_k = top_k

    def get_relevant_documents(self, query):
        if not self._docs or not self._embeddings:
            return []
        query_emb = embedding_model.embed_query(query)
        sims = np.array([
            np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            for emb in self._embeddings
        ])
        top_idx = sims.argsort()[-self._top_k:][::-1]
        return [self._docs[i] for i in top_idx]

retriever = InMemoryRetriever()
retriever.set_memory(st.session_state['documents'], st.session_state['embeddings'], top_k=5)

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

    chunked_docs = text_splitter.split_documents(new_docs)
    st.session_state['documents'].extend(chunked_docs)
    new_embeddings = embedding_model.embed_documents([doc.page_content for doc in chunked_docs])
    st.session_state['embeddings'].extend(new_embeddings)

    # Update retriever memory
    retriever.set_memory(st.session_state['documents'], st.session_state['embeddings'], top_k=5)

    st.success(f"Loaded {len(chunked_docs)} new chunks. Total in memory: {len(st.session_state['documents'])}")

# Get user query
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    relevant_docs = retriever.get_relevant_documents(user_input)
    if relevant_docs:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
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
for i in range(len(st.session_state['generated'])):
    message(st.session_state['generated'][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
