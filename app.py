"""
Main application entry point for the nodule detection service.
"""
import os
from api.routes import create_app
from utils.logging_config import logger

if __name__ == '__main__':
    try:
        # Create directories if they don't exist
        os.makedirs('temp_uploads', exist_ok=True)
        os.makedirs('temp_model', exist_ok=True)
        
        logger.info("Starting Flask server on port 5000...")
        app = create_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting Flask server: {str(e)}")
        import traceback
        traceback.print_exc()