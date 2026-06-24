import streamlit as st
import json
from agent import research_graph

st.set_page_config(
    page_title="Deep Research Console",
    page_icon="🤖",
    layout="centered"
)

# Custom minimalist styling to mirror premium chat UI frameworks
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    stDeployButton { display:none; }
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Deep Research Agent Cluster")
st.caption("Cyclical Multi-Agent State Graph Core (Planner ➔ Researcher ➔ Critic)")

# Initialize persistent message stream inside session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hydrate view with past message records
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user objectives
if user_query := st.chat_input("Enter your research objective..."):
    # Render user prompt instantly
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Initialize agent cluster runtime execution thread
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Orchestrating graph state memory nodes...*")
        
        try:
            # Inject payload natively into compiled local LangGraph execution thread
            initial_state = {
                "messages": [{"role": "user", "content": user_query}],
                "search_queries": [],
                "current_research": "",
                "loop_count": 0
            }
            
            # Execute state machine graph synchronously
            final_state = research_graph.invoke(initial_state)
            final_report = final_state.get("current_research", "⚠️ Architecture Error: Agent failed to compile research.")
            
            # Render compiled payload
            status_placeholder.markdown(final_report)
            st.session_state.messages.append({"role": "assistant", "content": final_report})
            
        except Exception as e:
            status_placeholder.error(f"Execution Error: {str(e)}")