"""
API route definitions for the nodule detection application.
"""
import os
import io
import numpy as np
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.logging_config import logger
from utils.image_utils import allowed_file
from model.loader import load_model
from model.inference import process_image
from visualization.visualizer import create_single_slice_visualization
from api.response_utils import get_progress, set_progress, create_error_response
from config import UPLOAD_FOLDER

# Initialize global model
model = None

def create_app():
    """
    Create and configure the Flask app.
    
    Returns:
        Configured Flask app
    """
    try:
        app = Flask(__name__)
        CORS(app)
        
        # Configuration
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Load model in a background thread to avoid blocking app startup
        def background_model_load():
            global model
            try:
                logger.info("Starting model loading in background thread...")
                model = load_model()
                logger.info("Background model loading completed successfully")
            except Exception as e:
                logger.error(f"Error in background model loading: {str(e)}")
        
        # Start background loading thread
        if model is None:
            thread = threading.Thread(target=background_model_load)
            thread.daemon = True  # Thread will exit when main program exits
            thread.start()
            logger.info("Background model loading initiated")
        
        # Keep the endpoint for explicit model loading if needed
        @app.route('/load_model', methods=['GET'])
        def force_load_model():
            """Endpoint to explicitly load the model or check loading status"""
            global model
            try:
                if model is None:
                    return jsonify({
                        'status': 'loading', 
                        'message': 'Model is still loading in background. Try again later.'
                    })
                else:
                    return jsonify({
                        'status': 'success', 
                        'message': 'Model is loaded and ready to use'
                    })
            except Exception as e:
                logger.error(f"Error checking model status: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint to check if service is running and model status"""
            global model
            return jsonify({
                'status': 'healthy', 
                'model_loaded': model is not None,
                'ready_for_inference': model is not None
            })
        
        @app.route('/progress', methods=['GET'])
        def progress_endpoint():
            """Endpoint to get current processing progress"""
            return jsonify({'progress': get_progress()})
        
        @app.route('/predict_multiple', methods=['POST'])
        def predict_multiple():
            """Endpoint to process multiple images and return individual visualizations"""
            try:
                global model
                set_progress(0)
                
                # Check if model is loaded yet
                if model is None:
                    logger.warning("Model not loaded yet, request received too early")
                    return jsonify({
                        'error': 'Model is still loading. Please try again in a few moments.'
                    }), 503  # Service Unavailable
                
                if 'files[]' not in request.files:
                    return create_error_response('No files uploaded', 400)
                
                files = request.files.getlist('files[]')
                if not files or files[0].filename == '':
                    return create_error_response('No files selected', 400)
                
                total_files = len(files)
                results = []
                
                # Process each file
                for index, file in enumerate(sorted(files, key=lambda x: x.filename)):
                    if not allowed_file(file.filename):
                        return create_error_response(
                            f'Invalid file type for {file.filename}. Only .npy files allowed', 
                            400
                        )
                    
                    try:
                        # Update progress
                        set_progress(int((index / total_files) * 100))
                        logger.info(f"Processing file {index+1}/{total_files}: {file.filename}")
                        
                        # Load and process the image
                        image_array = np.load(io.BytesIO(file.read()))
                        
                        # Normalize if needed
                        if image_array.max() > 1:
                            image_array = image_array / image_array.max()
                        
                        # Get prediction
                        prediction = process_image(image_array, model)
                        
                        # Create visualization with radiomic features
                        viz_result = create_single_slice_visualization(image_array, prediction)
                        
                        # Store results
                        results.append({
                            "filename": file.filename,
                            "visualization": viz_result["visualization"],
                            "nodules": viz_result["nodules"]
                        })
                        
                        logger.info(f"Completed processing file {file.filename} with {len(viz_result['nodules'])} nodules")
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return create_error_response(f'Error processing {file.filename}: {str(e)}')
                
                set_progress(100)
                logger.info(f"Successfully processed {total_files} files")
                return jsonify({'results': results})
                
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                set_progress(0)
                return create_error_response(str(e))
        
        return app
    
    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}")
        import traceback
        traceback.print_exc()
        raise