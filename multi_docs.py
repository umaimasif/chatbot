import os
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from load_doc import load_docs
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message # pip install streamlit_chat




load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")


llm_model = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(temperature=0.0, model=llm_model)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#### === packages to install ====
# pip install langchain pypdf openai chromadb tiktoken docx2txt

# load files
documents = load_docs()
chat_history = []

# Now we split the data into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=10
)
docs = text_splitter.split_documents(documents)

# create our vector db chromadb
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory='./data'
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)


#==== Streamlit front-end ====
st.title("Docs QA Bot using Langchain")
st.header("Ask anything about your documents... 🤖")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_query():
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text


# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    chat_history.append((user_input, result['answer']))  # Store for memory
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])

# Display chat
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')