"""Session state management."""

import streamlit as st
from typing import List, Dict, Any


def init_session_state():
    """Initialize session state."""
    defaults = {
        "messages": [],
        "model_loaded": False,
        "generating": False,
        "system_prompt": "",
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_messages() -> List[Dict[str, str]]:
    return st.session_state.get("messages", [])


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def clear_messages():
    st.session_state.messages = []


def set_generating(value: bool):
    st.session_state.generating = value


def is_generating() -> bool:
    return st.session_state.get("generating", False)
