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

# Import libraries after installation
import streamlit as st
import pandas as pd
import openai
import plotly.graph_objects as go
import traceback  # To capture and display detailed error traces

# Request OpenAI API Key from the User
st.sidebar.title("API Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Initialize OpenAI API Key
client = OpenAI(api_key=api_key)

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

# Step 3: Initial Role Selection
roles = ["Stakeholder", "Analyst", "Task Owner", "Project Manager", "Sponsor"]
selected_roles = st.multiselect("Select Required Roles:", roles)

# Step 4: AI-Powered RACI Suggestions
if st.button("Identify Key People"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for planning effective meetings."},
                {"role": "user", "content": f"Suggest key attendees for a {meeting_type} meeting with the objective: {meeting_objective}. Consider these roles: {', '.join(selected_roles)}."}
            ],
            max_tokens=200
        )
        ai_suggestions = response["choices"][0]["message"]["content"].strip().split("\n")
        st.markdown("### Key People Identified by AI")
        st.write(ai_suggestions)
    except Exception as e:
        ai_suggestions = []  # Ensure it's initialized even if the API call fails
        st.error(f"Failed to retrieve AI suggestions: {e}")

    # Ensure ai_suggestions is defined before using it
    suggested_roles = [{"Role": role, "Explanation": explanation} 
                       for suggestion in ai_suggestions
                       if (role := suggestion.split(":")[0].strip()) 
                       and (explanation := ":".join(suggestion.split(":")[1:]).strip())]
    st.dataframe(pd.DataFrame(suggested_roles))

# Step 5: Option to Remove People
if "suggested_roles" in st.session_state:
    st.markdown("### Modify Attendees")

    # Allow users to select roles to remove
    roles_to_remove = st.multiselect(
        "Select Roles to Remove:", 
        [person["Role"] for person in st.session_state["suggested_roles"]]
    )

    # Remove selected roles
    if st.button("Update Attendees"):
        st.session_state["suggested_roles"] = [
            person for person in st.session_state["suggested_roles"]
            if person["Role"] not in roles_to_remove
        ]
        st.success("Attendees updated.")

    # Show updated attendees
    st.markdown("### Final Attendee List")
    st.dataframe(pd.DataFrame(st.session_state["suggested_roles"]))

# Step 6: Visualize Meeting Room
if st.checkbox("Show Meeting Room Layout"):
    if "suggested_roles" in st.session_state:
        attendee_roles = [person["Role"] for person in st.session_state["suggested_roles"]]
    else:
        attendee_roles = roles

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1] * len(attendee_roles),
        theta=attendee_roles,
        mode='markers',
        marker=dict(size=10)
    ))
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise")))
    st.plotly_chart(fig)

# Step 7: Role Play Simulation (Placeholder)
if st.button("Start Role-Play Simulation"):
    st.markdown("### Role-Play Simulation")
