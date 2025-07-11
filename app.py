import streamlit as st
from inference_engine import generate_response  

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🧠 Classroom AI Assistant - By Team Chimera")
st.write("Ask me any academic question!")

user_input = st.text_input("Enter your question here:", key="user_input")

if user_input:
    st.markdown("**Assistant:**")
    response_placeholder = st.empty()
    full_response = ""

    for token in generate_response(user_input):
        full_response += token
        response_placeholder.markdown(full_response + "▌")

    response_placeholder.markdown(full_response)
