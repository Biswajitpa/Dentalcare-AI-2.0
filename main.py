import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables load up front
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from dental_agent.agent import dental_graph

# =====================================================================
# 1. PREMIUM BRANDING & WEB UI OVERLAYS
# =====================================================================
st.set_page_config(
    page_title="DentalOS Hub | Multi-Agent Practice Management",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI styling injection
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stChatMessage { border-radius: 12px; margin-bottom: 10px; }
        .hint-card { 
            background-color: #1e293b; 
            padding: 12px; 
            border-radius: 8px; 
            border-left: 4px solid #3b82f6;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. DYNAMIC SIDEBAR DATA LEDGER (Recruiter Wow Factor)
# =====================================================================
with st.sidebar:
    st.title("🦷 DentalOS Live View")
    st.caption("Enterprise Multi-Agent Sync Matrix")
    st.markdown("---")
    
    st.subheader("📋 Real-Time Slot Registry")
    try:
        # Pull live availability directly from data storage file
        df = pd.read_csv("doctor_availability.csv")
        df_display = df.copy()
        df_display.columns = ["Time Slot", "Specialization", "Doctor", "Available", "Patient ID"]
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Available": st.column_config.CheckboxColumn("Available"),
                "Patient ID": st.column_config.NumberColumn("Patient ID", format="%d")
            }
        )
        
        # Real-time metrics visualization panel
        total_slots = len(df)
        open_slots = int((df['is_available'].astype(str).str.strip().str.lower() == 'true').sum())
        booked_slots = total_slots - open_slots
        
        col1, col2 = st.columns(2)
        col1.metric("Available Slots", open_slots)
        col2.metric("Total Booked", booked_slots)
        
    except Exception as e:
        st.error(f"Ledger Pipeline Offline: {str(e)}")

    st.markdown("---")
    st.markdown("### 💡 System Architecture Note")
    st.info("This system uses standard LangGraph streaming callbacks. Token events are intercepted directly from the state machine pipeline chunks and dynamically updated on the screen.")

# =====================================================================
# 3. INTERACTIVE CHAT ENGINE (Ported cleanly from your loop)
# =====================================================================
st.markdown("## 🧠 Clinical Multi-Agent Routing Portal")

# Use st.session_state to persist your history array across button-press refreshes
if "history" not in st.session_state:
    st.session_state.history = []

# Display quick sample action shortcuts to hiring managers
with st.expander("🚀 View Production Prompt Templates", expanded=False):
    st.markdown("""
    <div class="hint-card"><b>Query Availability:</b> Show available slots for an orthodontist</div>
    <div class="hint-card"><b>Book:</b> Book patient 1000082 with Emily Johnson on 5/10/2026 9:00</div>
    <div class="hint-card"><b>Cancel:</b> Cancel appointment for patient 1000082 at 5/10/2026 9:00</div>
    <div class="hint-card"><b>Reschedule:</b> Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00</div>
    """, unsafe_allow_html=True)

# Render conversation logs explicitly from state history
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Intercept user chat inputs
if user_input := st.chat_input("Ask about schedules, bookings, or cancellations..."):
    
    # Display human input block onto screen layer instantly
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Maintain parity with your history tracking code block
    st.session_state.history.append(HumanMessage(content=user_input))
    
    # Render assistant placeholder to capture streaming text chunks live
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_streamed_text = ""
        final_messages = None
        
        try:
            # We port your EXACT streaming loop right here:
            for event_type, data in dental_graph.stream(
                {"messages": st.session_state.history},
                stream_mode=["messages", "values"],
                config={"recursion_limit": 20},
            ):
                if event_type == "messages":
                    chunk, meta = data
                    # Your identical conditions for token parsing
                    if (
                        isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and not getattr(chunk, "tool_calls", None)
                    ):
                        full_streamed_text += chunk.content
                        # Stream the tokens typing live into the web box frame!
                        response_placeholder.markdown(full_streamed_text + "▌")
                        
                elif event_type == "values":
                    final_messages = data.get("messages", [])
            
            # Fix final render layout state 
            response_placeholder.markdown(full_streamed_text if full_streamed_text else "Workflow complete!")
            st.toast("Database updated!", icon="💾")
            
        except Exception as exc:
            st.error(f"Error encountered: {exc}")
            # Maintain match logic with your exception handling block
            st.session_state.history.pop()
            final_messages = None
            
        # If state finalized successfully, update session trace and reload table views instantly
        if final_messages:
            st.session_state.history = final_messages
            st.rerun()