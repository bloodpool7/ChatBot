import streamlit as st
import Chatbot_Utils as utils
import os
import numpy as np 
import pandas as pd

openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]
namespace = "Policies"

st.set_page_config(page_title="Complete csvs")

file = st.file_uploader("Upload csv to complete", type = "csv")

if file:
    df = pd.read_csv(file)

    answers = []
    chat_history = []
    for question in df["question"]:
        result = utils.query_documents(question, namespace, openai_api_key = openai_api_key, pinecone_api_key = pinecone_api_key, pinecone_env=pinecone_env, index_name = "test", chat_history = chat_history) 
        answers.append(result['answer'])
        chat_history.append((question, result['answer']))
    
    df["answer"] = answers
    st.dataframe(df, width = 500, height = 500)
    st.download_button("Download csv", data = df.to_csv(), file_name = file.name[:-4] + "_completed.csv", mime = "text/csv")

