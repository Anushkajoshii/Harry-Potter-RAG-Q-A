import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if not groq_api_key or not google_api_key:
    st.error("API keys are not set. Please check your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# Gemma Model Document Q&A

st.title("RAG LLM - Harry Potter Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./harryPotter")  # Ensure this directory exists and contains PDFs
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Creation

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.time() - start_time
    
    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(response['answer'])

    # streamlit expander
    with st.expander("Document Similarity Search"):
        # relevant chunks
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("-------XXXXXXXXXXXXXXXXX-------")

# Footer section
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px;
        background-color: #f1f1f1;
        text-align: center;
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
        Made by Anushka Joshi
    </div>
    """, unsafe_allow_html=True
)
