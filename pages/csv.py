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

    