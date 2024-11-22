# Meeting Gen AI for StreamLit
import subprocess
import sys
import os

# Upgrade pip and install requirements
def setup_environment():
    try:
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip upgraded successfully.")
        
        # Check if requirements.txt exists
        if os.path.exists("requirements.txt"):
            print("Installing dependencies from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("Error: requirements.txt not found. Please provide a valid requirements file.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

# Set up the environment
setup_environment()

# Import libraries after installation
import streamlit as st
import pandas as pd
import openai
import plotly.graph_objects as go

# Initialize OpenAI API Key
openai.api_key = st.secrets["openai_api_key"]

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

# Step 4: AI-Powered Suggestions for Key People
if st.button("Identify Key People"):
    if selected_roles:
        # Generate suggestions for key attendees using OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"For a {meeting_type} meeting with the objective: {meeting_objective}, and roles: {', '.join(selected_roles)}, who are the key attendees? Provide a brief explanation for each role.",
            max_tokens=200
        )
        ai_suggestions = response.choices[0].text.strip().split("\n")
        
        # Display suggestions
        st.markdown("### Key People Identified by AI")
        st.write("\n".join(ai_suggestions))
        
        # Extract roles and names for modification
        suggested_roles = [{"Role": role, "Explanation": explanation} 
                           for suggestion in ai_suggestions 
                           if (role := suggestion.split(":")[0].strip()) 
                           and (explanation := ":".join(suggestion.split(":")[1:]).strip())]

        # Display suggested attendees in a table format
        st.session_state["suggested_roles"] = suggested_roles
        st.dataframe(pd.DataFrame(suggested_roles))
    else:
        st.warning("Please select at least one role to proceed.")

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
    st.write("This will simulate a conversation between the meeting attendees. (Coming soon!)")
