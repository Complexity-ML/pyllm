"""Custom styles for PyLLM UI."""

import streamlit as st


def apply_styles():
    """Apply custom CSS."""
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp {
        background-color: #09090b;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Message bubbles */
    .user-message {
        background: #1e40af;
        padding: 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
    }

    .assistant-message {
        background: #27272a;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        margin-right: 20%;
    }

    .message-role {
        font-size: 0.75rem;
        color: #a1a1aa;
        margin-bottom: 0.25rem;
    }

    .message-content {
        color: #fafafa;
        white-space: pre-wrap;
    }

    /* Input area */
    .stTextArea textarea {
        background-color: #18181b;
        border: 1px solid #27272a;
        color: #fafafa;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
    }

    .stButton > button:hover {
        background-color: #2563eb;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #18181b;
    }

    ::-webkit-scrollbar-thumb {
        background: #3f3f46;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
