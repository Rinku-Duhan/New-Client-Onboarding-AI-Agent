

import streamlit as st
import os

# --- BRIDGE: Load Secrets into Environment Variables ---
# This loop manually loads the keys from Streamlit secrets
# into the system environment so libraries like LangChain 
# can find them automatically.
for key in st.secrets:
    os.environ[key] = st.secrets[key]

#Use this part if you are using OPENAPI key and add the key in .env
# from data.employees import generate_employee_data
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# import logging
# from assistant import Assistant
# from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
# from langchain_groq import ChatGroq
# from gui import AssistantGUI


# if __name__ == "__main__":

#     load_dotenv()

#     logging.basicConfig(level=logging.INFO)

#     st.set_page_config(page_title="Umbrella Onboarding", page_icon="☂", layout="wide")

#     @st.cache_data(ttl=3600, show_spinner="Loading Employee Data...")
#     def get_user_data():
#         return generate_employee_data(1)[0]

#     @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
#     def init_vector_store(pdf_path):
#         try:
#             loader = PyPDFLoader(pdf_path)
#             docs = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=2000, chunk_overlap=200
#             )
#             splits = text_splitter.split_documents(docs)

#             embedding_function = OpenAIEmbeddings()
#             persistent_path = "./data/vectorstore"

#             vectorstore = Chroma.from_documents(
#                 documents=splits,
#                 embedding=embedding_function,
#                 persist_directory=persistent_path,
#             )

#             return vectorstore
#         except Exception as e:
#             logging.error(f"Error initializing vector store: {str(e)}")
#             st.error(f"Failed to initialize vector store: {str(e)}")
#             return None

#     customer_data = get_user_data()
#     vector_store = init_vector_store("data/umbrella_corp_policies.pdf")

#     if "customer" not in st.session_state:
#         st.session_state.customer = customer_data
#     if "messages" not in st.session_state:
#         st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

#     llm = ChatGroq(model="llama-3.1-8b-instant")

#     assistant = Assistant(
#         system_prompt=SYSTEM_PROMPT,
#         llm=llm,
#         message_history=st.session_state.messages,
#         employee_information=st.session_state.customer,
#         vector_store=vector_store,
#     )
    
#     gui = AssistantGUI(assistant)
#     gui.render()

#use this part of code if you are using Google API Key
from data.employees import generate_employee_data
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import logging
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq
from gui import AssistantGUI
import os


if __name__ == "__main__":

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Umbrella Onboarding", page_icon="☂", layout="wide")

    @st.cache_data(ttl=3600, show_spinner="Loading Employee Data...")
    def get_user_data():
        return generate_employee_data(1)[0]

    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            # ✅ Use Gemini embeddings instead of OpenAI
            embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            

            persistent_path = "./data/vectorstore"

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory=persistent_path,
            )

            return vectorstore
        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")
            return None

    customer_data = get_user_data()
    vector_store = init_vector_store("data/umbrella_corp_policies.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    # ✅ Groq for chat model (this is fine)
    llm = ChatGroq(model="llama-3.1-8b-instant")

    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        employee_information=st.session_state.customer,
        vector_store=vector_store,
    )
    
    gui = AssistantGUI(assistant)
    gui.render()
