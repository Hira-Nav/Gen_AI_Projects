#llm-quickstart model https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
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

    def save_audio(self):
        if not self.audio_frames:
            return None

        # Combine audio frames into a WAV file
        audio_data = np.concatenate(self.audio_frames, axis=0).astype(np.int16)
        audio_segment = AudioSegment(
            data=audio_data.tobytes(),
            sample_width=2,
            frame_rate=48000,
            channels=1,
        )
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_segment.export(temp_audio_file.name, format="wav")
        return temp_audio_file.name

# Choice: Record Audio or Upload File
option = st.radio("Select Input Method", ["Record Audio", "Upload Audio File"])

if option == "Record Audio":
    # WebRTC streamer setup
    webrtc_ctx = webrtc_streamer(
        key="audio-record",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=SpeechToTextProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx and webrtc_ctx.audio_processor:
        if st.button("Stop and Transcribe"):
            # Save and transcribe audio
            audio_file_path = webrtc_ctx.audio_processor.save_audio()
            if audio_file_path:
                st.success("Audio recorded successfully!")
                st.audio(audio_file_path, format="audio/wav")

                # Transcription using Whisper
                transcription = whisper_model.transcribe(audio_file_path)
                st.text_area("Transcription:", transcription.get("text", ""))
                os.unlink(audio_file_path)  # Clean up temporary file
            else:
                st.error("No audio was recorded.")

elif option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a WAV file:", type=["wav"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_path = temp_audio_file.name

        st.success("File uploaded successfully!")
        st.audio(temp_audio_path, format="audio/wav")

        # Transcribe uploaded audio using Whisper
        transcription = whisper_model.transcribe(temp_audio_path)
        st.text_area("Transcription:", transcription.get("text", ""))
        os.unlink(temp_audio_path)  # Clean up temporary file

# LangChain Integration for Transcriptions
if st.button("Send Transcription to Chat"):
    # Get the transcription text
    if option == "Record Audio" and webrtc_ctx and webrtc_ctx.audio_processor:
        transcription_text = webrtc_ctx.audio_processor.save_audio()
    elif option == "Upload Audio File" and uploaded_file is not None:
        transcription_text = transcription.get("text", "")
    else:
        st.warning("No transcription available!")
        transcription_text = None

    if transcription_text:
        # Add transcription to chat
        st.session_state.messages.append({"role": "user", "content": transcription_text})
        with st.chat_message("user"):
            st.markdown(transcription_text)

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
