import streamlit as st
import Chatbot_Utils as utils
import os

openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]
namespace = "Policies"

chat_history = []

st.set_page_config(page_title="Query PDFs")

st.write("Enter your query here:")
query = st.text_input("Query")

if query:
    result = utils.query_documents(query, namespace, openai_api_key = openai_api_key, pinecone_api_key = pinecone_api_key, pinecone_env=pinecone_env, index_name = "test", chat_history = chat_history) 
    chat_history.append((query, result['answer']))
    st.write(result['answer'])
