"""
Snowflake initialization utilities for RagaVani

This module provides utilities for initializing the application in a Snowflake environment.
"""

import os
import logging
import json

logger = logging.getLogger(__name__)

def is_running_in_snowflake():
    """Check if the application is running in Snowflake"""
    # Check for Snowflake-specific environment variables
    return os.environ.get('SNOWFLAKE_CONTAINER_SERVICE', '') == 'true'

def initialize_snowflake_environment():
    """Initialize the environment for running in Snowflake"""
    if not is_running_in_snowflake():
        logger.info("Not running in Snowflake, skipping Snowflake initialization")
        return
    
    logger.info("Initializing Snowflake environment")
    
    # Create data directory if it doesn't exist
    data_dir = "/data"
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Created data directory at {data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
    
    # Create user database file if it doesn't exist
    user_db_path = os.path.join(data_dir, "user_database.json")
    if not os.path.exists(user_db_path):
        try:
            default_user_email = os.environ.get('DEFAULT_USER_EMAIL', 'admin@ragavani.com')
            default_user_password = os.environ.get('DEFAULT_USER_PASSWORD', 'admin123')
            
            default_user_db = {
                "users": [
                    {
                        "email": default_user_email,
                        "password": default_user_password,
                        "role": "admin",
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                ]
            }
            
            with open(user_db_path, 'w') as f:
                json.dump(default_user_db, f, indent=2)
            
            logger.info(f"Created default user database at {user_db_path}")
        except Exception as e:
            logger.error(f"Failed to create user database: {e}")
    
    # Set environment variables for Snowflake
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    logger.info("Snowflake environment initialization complete")