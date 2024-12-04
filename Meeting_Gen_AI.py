# Meeting Gen AI for StreamLit

# Meeting Gen AI for StreamLit
import subprocess
import sys
import os

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Error installing {package}: {e}")

required_packages = [
    "streamlit",
    "pandas",
    "openai",
    "plotly"
]

for package in required_packages:
    try:
        __import__(package.split("==")[0])
    except ImportError:
        install_package(package)

import streamlit as st
import pandas as pd
import openai
import plotly.graph_objects as go
import traceback  # For detailed error logging

# Sidebar for API Key Input
st.sidebar.title("API Configuration")
openai.api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai.api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Title of the App
st.title("Meeting Agents - Gen AI Solution")
st.subheader("Plan Your Meeting with AI-Powered Suggestions")

# Step 1: Meeting Type Selection
meeting_type = st.selectbox(
    "Select Meeting Type:",
    ["Discovery", "Ideation", "Strategy", "Board Review", "Execution"]
)

# Step 2: Input Meeting Objective
meeting_objective = st.text_area(
    "Meeting Objective:",
    "Write a brief summary of your meeting's purpose."
)

# Step 3: Select Attendees
roles = ["Stakeholder", "Analyst", "Task Owner", "Project Manager", "Sponsor"]
selected_roles = st.multiselect("Select Required Roles:", roles)

# Step 4: AI-Powered Suggestions for Key People
if st.button("Identify Key People"):
    try:
        st.write("Calling OpenAI to identify key people...")
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant for planning effective meetings."},
                {"role": "user", "content": f"Suggest key attendees for a {meeting_type} meeting with the objective: {meeting_objective}. Consider these roles: {', '.join(selected_roles)}."}
            ],
            max_tokens=200
        )
        ai_suggestions = response.choices[0].message.content.strip().split("\n")
        st.markdown("### Key People Identified by AI")
        st.write(ai_suggestions)
    except Exception as e:
        ai_suggestions = []  # Initialize as empty if API call fails
        st.error(f"Failed to retrieve AI suggestions: {e}")
        st.write(traceback.format_exc())

    # Ensure ai_suggestions is defined before using it
    if ai_suggestions:
        suggested_roles = [{"Role": role, "Explanation": explanation} 
                           for suggestion in ai_suggestions
                           if (role := suggestion.split(":")[0].strip()) 
                           and (explanation := ":".join(suggestion.split(":")[1:]).strip())]
        st.markdown("### Suggested Roles")
        st.dataframe(pd.DataFrame(suggested_roles))
    else:
        st.warning("No suggestions were generated. Please check your input or try again.")

# Step 5: Visualize Meeting Room (Poker Table Style)
if st.checkbox("Show Meeting Room Layout"):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1] * len(roles),
        theta=roles,
        mode='markers',
        marker=dict(size=10)
    ))
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise")))
    st.plotly_chart(fig)

# Step 6: Role Play Simulation
if st.button("Start Role-Play Simulation"):
    st.markdown("### Role-Play Simulation")
    try:
        # Create a role-play conversation
        role_play_prompt = (
            f"Simulate a {meeting_type} meeting with the following objective: {meeting_objective}. "
            f"The attendees are {', '.join(selected_roles)}. Generate a dialogue showing their contributions."
        )

        st.write("Generating role-play simulation...")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a meeting simulation assistant."},
                {"role": "user", "content": role_play_prompt}
            ],
            max_tokens=500
        )
        
        role_play_output = response.choices[0].message.content.strip().split("\n")
        st.markdown("### Simulated Meeting Dialogue")
        st.write(role_play_output)
    except Exception as e:
        st.error(f"Failed to simulate role-play: {e}")
        st.write(traceback.format_exc())


import streamlit as st
import openai
import random

# Sidebar for API Key Input
st.sidebar.title("API Configuration")
openai.api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai.api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Title of the App
st.title("Interactive Meeting Room Simulator")
st.markdown("""
<style>
    .meeting-room {
        position: relative;
        width: 600px;
        height: 600px;
        background-image: url('https://media.istockphoto.com/id/508209880/photo/business-team-meeting-connection-digital-technology-concept.jpg?s=612x612&w=0&k=20&c=GB4Qw5a12M8rBx1QKmfvgdAgzvqSifx4cP3GvyQoWB8='); /* Replace with your own table image URL */
        background-size: cover;
        border-radius: 50%;
        margin: auto;
    }
    .seat {
        position: absolute;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #ddd;
        text-align: center;
        line-height: 50px;
        font-size: 14px;
        cursor: pointer;
    }
    .tooltip {
        visibility: hidden;
        position: absolute;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        font-size: 12px;
        z-index: 1;
    }
    .seat:hover .tooltip {
        visibility: visible;
    }
</style>
""", unsafe_allow_html=True)

# Input for meeting type and roles
meeting_type = st.selectbox("Select Meeting Type:", ["Discovery", "Strategy", "Board Review"])
meeting_objective = st.text_area("Meeting Objective:", "Write the purpose of your meeting.")
roles = st.multiselect("Invite Participants:", ["Stakeholder", "Analyst", "Task Owner", "Project Manager", "Sponsor"])

# Generate seat positions dynamically
if roles:
    st.markdown('<div class="meeting-room">', unsafe_allow_html=True)

    angle_step = 360 / len(roles)
    for i, role in enumerate(roles):
        angle = angle_step * i
        x = 250 + 200 * (random.random() - 0.5)
        y = 250 + 200 * (random.random() - 0.5)

        # Generate dynamic OpenAI responses
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {role}, contributing to a {meeting_type} meeting."},
                {"role": "user", "content": f"What would your contribution be to this objective: {meeting_objective}?"}
            ],
            max_tokens=150
        )
        ai_response = response.choices[0].message.content

        # Add seats to the table
        seat_html = f"""
        <div class="seat" style="top:{y}px;left:{x}px;">
            {role}
            <div class="tooltip">{ai_response}</div>
        </div>
        """
        st.markdown(seat_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.write("Hover over or click on a seat to see the participant's contribution.")

