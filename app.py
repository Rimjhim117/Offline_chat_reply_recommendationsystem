import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from ChatRec_Model import ChatRecommendationSystem

# Page Configuration
st.set_page_config(
    page_title="Chat Reply Recommender",
    layout="wide"
)

# Custom Elegant CSS Styles - Premium Light Theme
st.markdown("""
<style>
    /* Primary light theme styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling with premium blue/purple gradient */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2.5rem;
        text-align: center;
    }
    
    /* Premium card design (White background with soft shadow) */
    .recommendation-card {
        background: #ffffff;
        border-left: 5px solid #4f46e5;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.08);
    }
    .recommendation-rank {
        font-weight: 700;
        color: #4f46e5;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .recommendation-reply {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        background: rgba(79, 70, 229, 0.07);
        color: #4f46e5;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .context-box {
        background: #f1f5f9;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.6rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Application Heading
st.markdown("<h1 class='main-title'>Chat Reply Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Get real-time response recommendations for your conversation</p>", unsafe_allow_html=True)

# Expander for custom dataset upload
with st.expander("Load Custom Chat Data", expanded=False):
    st.write("Upload a CSV or Excel file containing columns: `Conversation ID`, `Sender`, `Message`, and optionally `Timestamp`.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# Load dataset
@st.cache_data
def load_data(file_obj):
    if file_obj.name.endswith('.xlsx'):
        df = pd.read_excel(file_obj)
    else:
        df = pd.read_csv(file_obj)
    return df

# Main Data Selection logic
data_path = 'conversation_data.csv'
data_loaded = False
df = None

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        data_loaded = True
        st.success(f"Successfully loaded uploaded dataset: **{uploaded_file.name}** ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
elif os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading default conversation_data.csv: {e}")

if not data_loaded or df is None:
    st.error("Error: No data available. Please place conversation_data.csv in the directory or upload a file.")
    st.stop()

# Helper to check if active dataset has changed
dataset_identifier = uploaded_file.name if uploaded_file else "default_csv"

# Initialize or re-train model if dataset changes
if 'chat_system' not in st.session_state or st.session_state.get('active_dataset') != dataset_identifier:
    with st.spinner("Training model on dataset... Please wait."):
        try:
            # Save a temp file to load it inside the class
            temp_path = "temp_chat_data.csv"
            df.to_csv(temp_path, index=False)
            
            system = ChatRecommendationSystem(use_semantic=True, semantic_weight=0.7)
            system.context_window = 3
            system.load_and_preprocess_data(temp_path)
            system.train()
            
            st.session_state.chat_system = system
            st.session_state.active_dataset = dataset_identifier
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            st.error(f"Failed to train model: {e}")
            st.stop()

system = st.session_state.chat_system

st.markdown("### Simulator")
st.write("Type a message or select a conversation sequence to see the recommended replies.")

# Layout: Simple inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Input Message")
    
    # Simple context helper dropdown
    context_option = st.selectbox(
        "Quick-Load Conversation Context",
        ["[No Context / Start Conversation]"] + [f"Conversation {cid}" for cid in df['Conversation ID'].unique()]
    )
    
    context_text = ""
    default_input = ""
    
    if context_option != "[No Context / Start Conversation]":
        cid = int(context_option.split()[-1])
        conv_msgs = df[df['Conversation ID'] == cid].sort_values('Timestamp')
        
        # Format previous messages as context
        context_list = []
        for _, row in conv_msgs.tail(4).iterrows():
            context_list.append(f"{row['Sender']}: {row['Message']}")
        
        if len(context_list) > 1:
            context_text = " [SEP] ".join(context_list[:-1])
            default_input = conv_msgs.iloc[-1]['Message']
            
    custom_context = st.text_area(
        "Conversation History (turns separated by [SEP])",
        value=context_text,
        placeholder="User B: Hi! [SEP] User A: Hey there! [SEP] User B: How are you?",
        help="Paste or edit previous conversation turns here to guide context-aware recommendations."
    )
    
    user_b_msg = st.text_input(
        "Trigger Message (User B)",
        value=default_input,
        placeholder="Type a message here..."
    )

with col2:
    st.markdown("#### Recommended Replies (User A)")
    
    if user_b_msg:
        predictions = system.predict_reply(
            user_b_message=user_b_msg,
            context=custom_context,
            top_k=3
        )
        
        for idx, pred in enumerate(predictions, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-rank">Recommendation #{idx}</div>
                <div class="recommendation-reply">"{pred['reply']}"</div>
                <div>
                    <span class="score-badge">Match Confidence: {pred['similarity'] * 100:.1f}%</span>
                </div>
                <div class="context-box">
                    <strong>Based on historical trigger:</strong> "{pred['original_input']}"<br/>
                    <strong>Matched Context:</strong> {pred['context'] if pred['context'] else '<i>None</i>'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Type a message on the left to see recommended replies.")
