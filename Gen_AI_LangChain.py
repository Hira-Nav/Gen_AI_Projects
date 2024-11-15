#llm-quickstart model https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import whisper
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import tempfile
import os


st.title('🦜🔗 ChatGPT-like Clone using LangChain')

# Sidebar for API key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Initialize ChatOpenAI model
if openai_api_key.startswith("sk-"):
    model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
else:
    st.warning("Please enter a valid OpenAI API key!", icon="⚠")
    st.stop()

# Whisper Model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# AudioProcessor for live microphone input
class SpeechToTextProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorder = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    def recv_audio(self, frame):
        self.recorder.write(frame.to_ndarray().tobytes())
        return frame

    def get_transcription(self):
        self.recorder.flush()
        result = whisper_model.transcribe(self.recorder.name)
        os.unlink(self.recorder.name)
        return result["text"]


# Checkbox to toggle start/stop recording
recording = st.checkbox("Start Recording", value=False)

if recording:
    # Stream live audio
    st.write("Recording in progress...")

    # WebRTC Streamer for live audio capture
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=SpeechToTextProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    
    # Display transcription as audio is processed
    if webrtc_ctx.audio_processor:
        transcription = webrtc_ctx.audio_processor.get_transcription()
        st.write(f"Live Transcription: {transcription}")
else:
    st.write("Recording is stopped.")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("hi, how can i help?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for LangChain Human and AI
    formatted_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # Generate assistant response using LangChain
    with st.chat_message("assistant"):
        response = model.predict_messages(formatted_messages)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        # Convert assistant response to audio
        tts = gTTS(response.content)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)
        st.audio(audio_file.name, format="audio/mp3")
