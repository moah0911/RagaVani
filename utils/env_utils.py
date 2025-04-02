"""
Environment utilities for RagaVani application

This module provides functions for loading and accessing environment variables.
"""

import os
from dotenv import load_dotenv
import logging

# Configure logger
logger = logging.getLogger(__name__)

def load_environment_variables():
    """
    Load environment variables from .env file
    
    Returns:
        bool: True if .env file was loaded, False otherwise
    """
    try:
        # Load environment variables from .env file
        # Set override=True to allow environment variables to override .env file values
        dotenv_loaded = load_dotenv(override=True)
        
        if dotenv_loaded:
            logger.info("Loaded environment variables from .env file")
        else:
            logger.info("No .env file found or it was empty, using environment variables")
            
        return dotenv_loaded
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False

def get_environment_variable(name, default=None):
    """
    Get environment variable value
    
    Parameters:
        name (str): Name of environment variable
        default (any, optional): Default value if environment variable is not found
    
    Returns:
        str: Value of environment variable, or default if not found
    """
    try:
        # Load environment variables if not already loaded
        if not hasattr(get_environment_variable, "_env_loaded") or not get_environment_variable._env_loaded:
            load_environment_variables()
            setattr(get_environment_variable, "_env_loaded", True)
            
        # Get environment variable value
        value = os.environ.get(name, default)
        
        # If value is not found and no default is provided, log a warning
        if value is None and default is None:
            logger.warning(f"Environment variable {name} not found and no default provided")
            
        return value
    except Exception as e:
        logger.error(f"Error getting environment variable {name}: {e}")
        return default