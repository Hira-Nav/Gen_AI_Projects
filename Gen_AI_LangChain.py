import streamlit as st
from langchain.llms import OpenAI

st.title('🦜🔗 Quickstart App')

# Sidebar for API key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Function to generate a response
def generate_response(input_text):
    if not openai_api_key or not openai_api_key.startswith('sk-'):
        st.error("Invalid or missing OpenAI API key.")
        return
    try:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        response = llm(input_text)
        st.info(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")

# Check if API key is valid
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
else:
    # Main form
    with st.form('my_form'):
        text = st.text_area('Enter text:', placeholder='Type your question or prompt here...')
        submitted = st.form_submit_button('Submit')
        if submitted:
            generate_response(text)
