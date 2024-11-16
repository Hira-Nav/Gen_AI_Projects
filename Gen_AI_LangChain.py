#llm-quickstart model https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import whisper
import tempfile
import os
import numpy as np
from pydub import AudioSegment

# App Title
st.title("🎙️ Live Speech-to-Text and Chat with LangChain")

# Sidebar for OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Validate API Key and Initialize LangChain Model
if openai_api_key.startswith("sk-"):
    model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
else:
    st.warning("Please enter a valid OpenAI API key!", icon="⚠")
    st.stop()

# Load Whisper Model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Audio Processor for Speech-to-Text
class SpeechToTextProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv_audio(self, frame):
        self.audio_frames.append(frame.to_ndarray())
        return frame

    def get_transcription(self):
        if not self.audio_frames:
            return None

        # Combine audio frames and save as WAV
        audio_data = np.concatenate(self.audio_frames, axis=0).astype(np.int16)
        audio_segment = AudioSegment(
            data=audio_data.tobytes(),
            sample_width=2,
            frame_rate=48000,
            channels=1,
        )
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_segment.export(temp_audio_file.name, format="wav")

        # Transcribe with Whisper
        transcription_result = whisper_model.transcribe(temp_audio_file.name)
        os.unlink(temp_audio_file.name)  # Cleanup temporary file
        return transcription_result.get("text", "")

# Initialize session state for chat and transcription
if "messages" not in st.session_state:
    st.session_state.messages = []
if "live_transcription" not in st.session_state:
    st.session_state.live_transcription = ""

# WebRTC Streamer for Real-Time Audio Input
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SpeechToTextProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Live Transcription Output
if webrtc_ctx and webrtc_ctx.audio_processor:
    transcription = webrtc_ctx.audio_processor.get_transcription()
    if transcription:
        st.session_state.live_transcription = transcription

    # Display live transcription dynamically
    st.text_area("Live Transcription:", st.session_state.live_transcription, height=150)

    # Send transcription to chat
    if st.button("Send Transcription to Chat"):
        st.session_state.messages.append({"role": "user", "content": st.session_state.live_transcription})
        with st.chat_message("user"):
            st.markdown(st.session_state.live_transcription)

        # Generate assistant response
        formatted_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]
        response = model.predict_messages(formatted_messages)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        # Display response
        with st.chat_message("assistant"):
            st.markdown(response.content)

        # Convert assistant response to audio
        tts = gTTS(response.content)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)
        st.audio(audio_file.name, format="audio/mp3")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Manual Text Input for Chat
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    formatted_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]
    response = model.predict_messages(formatted_messages)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    # Convert response to audio
    tts = gTTS(response.content)
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)
    st.audio(audio_file.name, format="audio/mp3")