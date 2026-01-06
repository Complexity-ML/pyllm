"""Sidebar component."""

import streamlit as st
import streamlit_shadcn_ui as ui

from pyllm.ui.api import get_client
from pyllm.ui.state import clear_messages


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## PyLLM Chat")

        # API status
        client = get_client()
        health = client.health()

        if health and health.get("model_loaded"):
            ui.badge(text="Connected", variant="default", key="status_badge")
        elif health:
            ui.badge(text="No Model", variant="secondary", key="status_badge")
        else:
            ui.badge(text="Offline", variant="destructive", key="status_badge")

        st.markdown("---")

        # Generation settings
        st.markdown("### Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key="temperature",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=256,
            step=64,
            key="max_tokens",
        )

        st.markdown("---")

        # System prompt
        st.markdown("### System Prompt")
        system_prompt = st.text_area(
            "System",
            key="system_prompt",
            height=100,
            placeholder="You are a helpful assistant...",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Actions
        if ui.button("Clear Chat", key="clear_btn", variant="outline"):
            clear_messages()
            st.rerun()

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
