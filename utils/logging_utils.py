"""
Logging Utilities for RagaVani application

This module provides logging functionality for tracking
application events, errors, and performance metrics.
"""

import logging
import time
import os
import json
from datetime import datetime

# Configure logging
def setup_logger():
    """Set up the application logger"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/ragavani.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create and return app logger
    logger = logging.getLogger("RagaVani")
    return logger

# Get application logger
def get_app_logger():
    """Get or create the application logger"""
    logger = logging.getLogger("RagaVani")
    if not logger.handlers:  # If logger is not yet configured
        return setup_logger()
    return logger

# Log user activity
def log_user_activity(activity_type, user=None, details=None):
    """
    Log user activity
    
    Parameters:
        activity_type (str): Type of activity
        user (str, optional): Username
        details (dict, optional): Additional details
    """
    logger = get_app_logger()
    
    # Create activity log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "activity_type": activity_type,
        "user": user,
        "details": details or {}
    }
    
    # Log to file
    logger.info(f"User Activity - {activity_type} - User: {user}")
    
    # Append to activity log file
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/user_activity.log"
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to activity log: {str(e)}")

# Log errors
def log_error(component, error, context=None):
    """
    Log application errors
    
    Parameters:
        component (str): Component where the error occurred
        error (Exception): The exception object
        context (dict, optional): Additional context information
    """
    logger = get_app_logger()
    
    # Log the error
    logger.error(f"Error in {component}: {str(error)}")
    
    if context:
        logger.error(f"Context: {json.dumps(context)}")
    
    # Create error log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "component": component,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    # Append to error log file
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/errors.log"
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to error log: {str(e)}")

# Performance metrics
_PERFORMANCE_METRICS = {
    "ai_response_times": [],
    "audio_processing_times": [],
    "page_load_times": []
}

def record_performance_metric(metric_type, operation, duration_ms, details=None):
    """
    Record a performance metric
    
    Parameters:
        metric_type (str): Type of metric (ai_response_times, audio_processing_times, etc.)
        operation (str): Specific operation being measured
        duration_ms (float): Duration in milliseconds
        details (dict, optional): Additional details
    """
    # Create metric entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "duration_ms": duration_ms,
        "details": details or {}
    }
    
    # Add to in-memory metrics
    if metric_type in _PERFORMANCE_METRICS:
        _PERFORMANCE_METRICS[metric_type].append(entry)
        
        # Keep only the most recent 100 entries
        if len(_PERFORMANCE_METRICS[metric_type]) > 100:
            _PERFORMANCE_METRICS[metric_type] = _PERFORMANCE_METRICS[metric_type][-100:]
    
    # Log to performance log file
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/performance_{metric_type}.log"
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger = get_app_logger()
        logger.error(f"Failed to write to performance log: {str(e)}")

def get_performance_metrics():
    """
    Get current performance metrics
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    return _PERFORMANCE_METRICS