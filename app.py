import os
import streamlit as st
from langchain.chains                               import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents             import create_stuff_documents_chain
from langchain_core.chat_history                    import BaseChatMessageHistory
from langchain_community.chat_message_histories     import ChatMessageHistory
from langchain_core.prompts                         import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma                               import Chroma
from langchain_groq                                 import ChatGroq
from langchain_huggingface                          import HuggingFaceEmbeddings
from langchain_text_splitters                       import RecursiveCharacterTextSplitter
from langchain_community.document_loaders           import PyPDFLoader
from langchain_core.runnables.history               import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## Streamlit application

st.title("Conversational RAG Application with PDF Uploads and Chat History")
st.write("Upload PDF and have a conversation!")

apiKey = st.text_input("Enter Your Groq API Key: ", type = "password")
if apiKey:
    llm = ChatGroq(groq_api_key = apiKey, model_name = "Gemma2-9b-It")
    