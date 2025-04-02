"""
Session Handler for RagaVani application

This module provides functionality for managing Streamlit session state.
"""

import streamlit as st
import json
import time

def initialize_session():
    """Initialize session state variables"""
    # Audio data
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None

    if "tanpura_audio" not in st.session_state:
        st.session_state.tanpura_audio = None

    if "tabla_audio" not in st.session_state:
        st.session_state.tabla_audio = None

    if "melody_audio" not in st.session_state:
        st.session_state.melody_audio = None

    if "neural_audio" not in st.session_state:
        st.session_state.neural_audio = None

    # Analysis results
    if "raga_analysis" not in st.session_state:
        st.session_state.raga_analysis = None

    if "tala_analysis" not in st.session_state:
        st.session_state.tala_analysis = None

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # History of actions
    if "history" not in st.session_state:
        st.session_state.history = []

def save_to_session(key, value):
    """
    Save a value to session state
    
    Parameters:
        key (str): Key for storing the value
        value (any): Value to store
    """
    st.session_state[key] = value

def get_from_session(key):
    """
    Get a value from session state
    
    Parameters:
        key (str): Key for retrieving the value
    
    Returns:
        any: The stored value, or None if not found
    """
    return st.session_state.get(key)

def add_to_history(action, details=None):
    """
    Add an action to the history
    
    Parameters:
        action (str): Description of the action
        details (dict, optional): Additional details
    """
    # Prepare entry
    entry = {
        "timestamp": time.time(),
        "action": action,
        "details": details or {}
    }
    
    # Add to history
    if "history" in st.session_state:
        st.session_state.history.append(entry)
    else:
        st.session_state.history = [entry]
    
    # Limit history size
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def clear_session():
    """Clear the session state"""
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Reinitialize
    initialize_session()