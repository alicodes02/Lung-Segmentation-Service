"""
Helper functions for API responses.
"""
from flask import jsonify
from utils.logging_config import logger

# Global variable to track processing progress
processing_progress = 0

def get_progress():
    """
    Get the current processing progress.
    
    Returns:
        Current processing progress as a percentage
    """
    global processing_progress
    return processing_progress

def set_progress(value):
    """
    Set the current processing progress.
    
    Args:
        value: Progress value as a percentage (0-100)
    """
    global processing_progress
    processing_progress = value

def create_error_response(message, status_code=500):
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (jsonified error response, status code)
    """
    logger.error(f"Error: {message}")
    return jsonify({'error': message}), status_code

def create_success_response(data, message="Success"):
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Jsonified success response
    """
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data
    })