"""Chat components."""

import streamlit as st
import streamlit_shadcn_ui as ui
from typing import Dict


def render_message(msg: Dict[str, str], index: int):
    """Render a chat message."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-role">You</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-role">Assistant</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div style="
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            padding: 0.5rem;
            font-style: italic;
        ">
            System: {content}
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    """Render chat input area."""
    col1, col2 = st.columns([6, 1])

    with col1:
        message = st.text_area(
            "Message",
            key="chat_input",
            height=80,
            placeholder="Type your message...",
            label_visibility="collapsed",
        )

    with col2:
        send = ui.button("Send", key="send_btn", variant="default")

    return message if send else None
