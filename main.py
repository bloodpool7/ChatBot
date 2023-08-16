import streamlit as st

st.set_page_config(page_title="Streamlit App", page_icon=":smiley:")
st.title("Streamlit App")

st.sidebar.success("Select a page to begin")

st.write("Select a page from the sidebar to begin. Before querying a csv or pdf, make sure to add it to the index.")

st.session_state['namespace'] = "testtwo"