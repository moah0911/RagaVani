"""
Healthcheck module for RagaVani

This module provides a simple healthcheck endpoint for the application
to ensure it's running properly in Snowflake.
"""

import streamlit as st
import json
import os
import platform
import sys
import time
from datetime import datetime

def render_healthcheck_page():
    """Render the healthcheck page"""
    # Hide the page from the sidebar
    st.set_page_config(page_title="RagaVani Healthcheck", page_icon="🎵")
    
    # Get system information
    system_info = {
        "application": "RagaVani",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "modules": {
            "streamlit": st.__version__,
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "librosa": __import__("librosa").__version__,
            "tensorflow": __import__("tensorflow").__version__
        }
    }
    
    # Check if data directory exists and is writable
    data_dir = "/data"
    if os.path.exists(data_dir):
        if os.access(data_dir, os.W_OK):
            system_info["data_directory"] = {
                "path": data_dir,
                "status": "writable"
            }
        else:
            system_info["data_directory"] = {
                "path": data_dir,
                "status": "not writable"
            }
            system_info["status"] = "warning"
    else:
        system_info["data_directory"] = {
            "path": data_dir,
            "status": "not found"
        }
        system_info["status"] = "warning"
    
    # Display the healthcheck information
    st.title("RagaVani Healthcheck")
    
    st.markdown(f"""
    ### Status: {system_info['status']}
    
    **Application**: RagaVani - Indian Classical Music Analysis & Synthesis  
    **Timestamp**: {system_info['timestamp']}  
    **Python Version**: {system_info['python_version']}  
    **Platform**: {system_info['platform']}  
    """)
    
    st.subheader("Module Versions")
    for module, version in system_info["modules"].items():
        st.markdown(f"**{module}**: {version}")
    
    st.subheader("Data Directory")
    st.markdown(f"""
    **Path**: {system_info['data_directory']['path']}  
    **Status**: {system_info['data_directory']['status']}  
    """)
    
    # Return the system info as JSON for API access
    return json.dumps(system_info)