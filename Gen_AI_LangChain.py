#llm-quickstart model https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart

import streamlit as st
from langchain_openai.chat_models import ChatOpenAI


st.title('🦜🔗 Quickstart App')

# Sidebar for API key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Initialize ChatOpenAI model
if openai_api_key.startswith("sk-"):
    model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
else:
    st.warning("Please enter a valid OpenAI API key!", icon="⚠")
    st.stop()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using LangChain
    with st.chat_message("assistant"):
        response = model.predict(
            messages=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ]
        )
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
