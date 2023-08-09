import streamlit as st
import Chatbot_Utils as utils
import os

openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]
namespace = "Policies"

st.set_page_config(page_title="Add PDFs")

if 'added_pdfs' not in st.session_state:
    st.session_state['added_pdfs'] = set()

files = st.file_uploader("Upload PDFs", type = "pdf", accept_multiple_files = True)
files_dict = {}

if files:
    current_set = set([utils.hash_pdf(file.getvalue()) for file in files])

    for file in files:
        files_dict[utils.hash_pdf(file.getvalue())] = file
    
    added = list(current_set - st.session_state['added_pdfs'])
    deleted = list(st.session_state['added_pdfs'] - current_set)  
    st.session_state['added_pdfs'] = current_set

    if len(added) > 0:
        added = [files_dict[file] for file in added]
        utils.add_pdfs(added, namespace, openai_api_key, pinecone_api_key, pinecone_env, "test")
    if len(deleted) > 0:
        utils.delete_pdfs(deleted, True, namespace, pinecone_api_key, pinecone_env, "test")

if st.button("Reset Pinecone"):
    utils.delete_pdfs([], False, namespace, pinecone_api_key, pinecone_env, "test")
    st.session_state['added_pdfs'] = set()