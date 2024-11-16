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
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    def recv_audio(self, frame):
        self.audio_frames.append(frame.to_ndarray())
        return frame

    def get_transcription(self):
        # Combine audio frames and save to a temporary WAV file
        from pydub import AudioSegment
        audio = AudioSegment(
            data=b"".join([frame.tobytes() for frame in self.audio_frames]),
            sample_width=2,
            frame_rate=48000,
            channels=1,
        )
        audio.export(self.temp_file.name, format="wav")

        # Transcribe using Whisper
        transcription_result = whisper_model.transcribe(self.temp_file.name)
        os.unlink(self.temp_file.name)  # Clean up temporary file
        return transcription_result["text"]

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# WebRTC Streamer for Real-Time Audio Input
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SpeechToTextProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Live Transcription and Interaction
if webrtc_ctx and webrtc_ctx.audio_processor:
    transcription = webrtc_ctx.audio_processor.get_transcription()
    if transcription:
        st.info(f"🎤 Transcription: {transcription}")

        # Add transcription to chat and process response
        if st.button("Send Transcription to Chat"):
            # Add user transcription to chat history
            st.session_state.messages.append({"role": "user", "content": transcription})
            with st.chat_message("user"):
                st.markdown(transcription)

            # Prepare formatted messages for LangChain
            formatted_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            # Generate LangChain response
            with st.chat_message("assistant"):
                response = model.predict_messages(formatted_messages)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

                # Convert response to audio
                tts = gTTS(response.content)
                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(audio_file.name)
                st.audio(audio_file.name, format="audio/mp3")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text Input for Manual Chat
if prompt := st.chat_input("Type your message..."):
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for LangChain
    formatted_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # Generate LangChain response
    with st.chat_message("assistant"):
        response = model.predict_messages(formatted_messages)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        # Convert assistant response to audio
        tts = gTTS(response.content)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)
        st.audio(audio_file.name, format="audio/mp3")