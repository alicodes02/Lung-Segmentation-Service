"""
Logging configuration for the nodule detection application.
"""
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Also log to file
    ]
)
logger = logging.getLogger('nodule_detection')