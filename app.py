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
    llm = ChatGroq(groq_api_key = apiKey, model_name="Gemma2-9b-It")
    sessionId = st.text_input("Session ID", value="default")
    if 'store' not in st.session_state:
        st.session_state.sotre = {}

    uploadFiles = st.file_uploader("Chose A PDF File: ", type="pdf", accept_multiple_files=False)
    if uploadFiles:
        documents = []
        tempPDF = "./temp.pdf"
        with open(tempPDF, "wb") as file:
            file.write(uploadFiles.getvalue())
            fileName = uploadFiles.name

        loader = PyPDFLoader(tempPDF)
        docs = loader.load()
        documents.extend(docs)

        textSplitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = textSplitter.split_documents(documents)
        vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorStore.as_retriever()

        contextualizeQuestionSystemPrompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualizeQuestionPrompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualizeQuestionSystemPrompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        historyAwareRetriever = create_history_aware_retriever(llm, retriever, contextualizeQuestionPrompt)

        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        questionPrompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        questionAnswerChain = create_stuff_documents_chain(llm, questionPrompt)
        rag_chain = create_retrieval_chain(historyAwareRetriever, questionAnswerChain)
