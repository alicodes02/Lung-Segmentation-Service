import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import torch
import os
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from matplotlib.patches import Rectangle
from werkzeug.utils import secure_filename
from Unet.unet_model import UNet
import traceback
# Import for radiomic features extraction
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

# Initialize model and global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
processing_progress = 0

# Initialize radiomics feature extractor
def init_radiomics_extractor():
    """Initialize the radiomics feature extractor with default settings"""
    try:
        # Set up the feature extractor with common settings
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # Use original spacing
        settings['interpolator'] = sitk.sitkBSpline
        settings['verbose'] = False
        
        # Initialize extractor with these settings
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Enable features (adjust as needed)
        extractor.enableFeatureClassByName('firstorder', True)
        
        # IMPORTANT: Enable shape2D instead of shape for 2D images
        extractor.enableFeatureClassByName('shape2D', True)  # Enable 2D shape features
        extractor.disableFeatureClassByName('shape', True)   # Disable 3D shape features
        
        extractor.enableFeatureClassByName('glcm', True)
        extractor.enableFeatureClassByName('glrlm', True)
        
        return extractor
    except Exception as e:
        print(f"Error initializing radiomics extractor: {str(e)}")
        traceback.print_exc()
        return None

def extract_radiomic_features(image, mask):
    """Extract radiomic features from a single nodule"""
    try:
        # Make sure we have enough pixels to analyze
        if np.sum(mask) < 10:
            print(f"Warning: Mask too small for reliable feature extraction: {np.sum(mask)} pixels")
            # Return basic features based on image properties
            return {
                "Area (pixels)": float(np.sum(mask)),
                "Perimeter": float(np.sum(mask) * 0.5),  # Approximation
                "Elongation": 1.0,  # Default
                "Sphericity": 0.8,  # Default
                "Mean Intensity": float(np.mean(image[mask > 0])) if np.sum(mask) > 0 else 0,
                "Maximum": float(np.max(image[mask > 0])) if np.sum(mask) > 0 else 0,
                "Minimum": float(np.min(image[mask > 0])) if np.sum(mask) > 0 else 0,
                "Range": float(np.ptp(image[mask > 0])) if np.sum(mask) > 0 else 0,
                "Variance": float(np.var(image[mask > 0])) if np.sum(mask) > 0 else 0,
                "GLCM Contrast": 1.0  # Default
            }
        
        # Convert numpy arrays to SimpleITK images - ensure 2D format
        image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.int32))
        
        # Initialize extractor if not done already
        extractor = init_radiomics_extractor()
        
        # Extract features
        features = extractor.execute(image_sitk, mask_sitk)
        
        # Process features for display
        feature_dict = {}
        
        # Select key features to display (can be expanded)
        key_features = {
            'firstorder_Mean': 'Mean Intensity',
            'firstorder_Energy': 'Energy',
            'firstorder_Entropy': 'Entropy',
            'firstorder_Maximum': 'Maximum',
            'firstorder_Minimum': 'Minimum',
            'firstorder_Range': 'Range',
            'firstorder_Kurtosis': 'Kurtosis',
            'firstorder_Skewness': 'Skewness',
            'firstorder_TotalEnergy': 'Total Energy',
            'firstorder_Variance': 'Variance',
            'shape2D_PixelSurface': 'Area (pixels)',  # Use shape2D features
            'shape2D_Perimeter': 'Perimeter',
            'shape2D_Elongation': 'Elongation',
            'shape2D_Sphericity': 'Sphericity',
            'glcm_Correlation': 'GLCM Correlation',
            'glcm_Contrast': 'GLCM Contrast',
            'glrlm_RunEntropy': 'GLRLM Run Entropy'
        }
        
        # Extract the desired features
        for key, display_name in key_features.items():
            if key in features:
                feature_dict[display_name] = float(features[key])
        
        print(f"Successfully extracted {len(feature_dict)} features")
        return feature_dict
    except Exception as e:
        print(f"Error extracting radiomic features: {str(e)}")
        traceback.print_exc()
        
        # Return basic features as fallback
        try:
            area = np.sum(mask)
            return {
                "Area (pixels)": float(area),
                "Mean Intensity": float(np.mean(image[mask > 0])) if area > 0 else 0,
                "Maximum": float(np.max(image[mask > 0])) if area > 0 else 0,
                "Minimum": float(np.min(image[mask > 0])) if area > 0 else 0
            }
        except:
            return {"Area (pixels)": 10.0}  # Absolute minimum fallback

def get_bounding_boxes(prediction, min_size=10):
    """Get bounding boxes around nodules"""
    try:
        # Label connected components in the prediction
        labeled_array, num_features = ndimage.label(prediction)
        boxes = []
        
        for i in range(1, num_features + 1):
            # Get coordinates for each labeled region
            coords = np.where(labeled_array == i)
            if len(coords[0]) < min_size:  # Skip very small regions
                continue
                
            y_min, y_max = int(np.min(coords[0])), int(np.max(coords[0]))
            x_min, x_max = int(np.min(coords[1])), int(np.max(coords[1]))
            
            # Add some padding around the box
            padding = 2
            y_min = max(0, y_min - padding)
            y_max = min(prediction.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(prediction.shape[1], x_max + padding)
            
            # Calculate centroid
            centroid_y = (y_min + y_max) // 2
            centroid_x = (x_min + x_max) // 2
            
            # Extract mask for this nodule
            nodule_mask = np.zeros_like(prediction)
            nodule_mask[labeled_array == i] = 1
            
            boxes.append({
                "coords": (x_min, y_min, x_max, y_max),
                "centroid": (centroid_x, centroid_y),
                "mask": nodule_mask
            })
        
        return boxes
    except Exception as e:
        print(f"Error in get_bounding_boxes: {str(e)}")
        traceback.print_exc()
        return []

def load_model():
    """Initialize the model"""
    try:
        print("Loading model...")
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        
        # Load checkpoint
        checkpoint = torch.load('model_outputs/UNET_base/checkpoint.pth', 
                              map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel state dict if necessary
        if 'module' in list(state_dict.keys())[0]:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'npy'}

def process_image(image_array, model):
    """Process a single image through the model"""
    try:
        image = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output)
        
        return output
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        traceback.print_exc()
        raise

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error in fig_to_base64: {str(e)}")
        traceback.print_exc()
        raise

def create_single_slice_visualization(original, prediction):
    """Create visualization for a single slice with layers"""
    try:
        plt.ioff()
        
        # Get bounding boxes for the prediction with enhanced info
        boxes_data = get_bounding_boxes(prediction)
        
        # Process radiomic features for each nodule
        nodule_data = []
        for i, box_info in enumerate(boxes_data):
            # Extract coordinates
            box = box_info["coords"]
            centroid = box_info["centroid"]
            
            # Get the mask for this specific nodule
            nodule_mask = box_info["mask"]
            
            # Extract radiomic features for this nodule
            features = extract_radiomic_features(original, nodule_mask)
            
            # Store all nodule information
            nodule_data.append({
                "id": i + 1,
                "box": {
                    "x_min": int(box[0]),
                    "y_min": int(box[1]),
                    "x_max": int(box[2]),
                    "y_max": int(box[3])
                },
                "centroid": {
                    "x": int(centroid[0]),
                    "y": int(centroid[1])
                },
                "features": features
            })
        
        # Original image
        fig_original = plt.figure(figsize=(5, 5))
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        original_base64 = fig_to_base64(fig_original)
        plt.close(fig_original)
        
        # Segmentation layer
        fig_seg = plt.figure(figsize=(5, 5))
        plt.imshow(prediction, cmap='hot', alpha=0.7)
        plt.axis('off')
        segmentation_base64 = fig_to_base64(fig_seg)
        plt.close(fig_seg)
        
        # Nodules layer (boxes)
        fig_nodules = plt.figure(figsize=(5, 5))
        plt.imshow(np.zeros_like(original), cmap='gray')
        for box_info in boxes_data:
            box = box_info["coords"]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height,
                           fill=False, color='cyan', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add centroid marker
            centroid_x, centroid_y = box_info["centroid"]
            plt.plot(centroid_x, centroid_y, 'ro', markersize=3)
            
        plt.axis('off')
        nodules_base64 = fig_to_base64(fig_nodules)
        plt.close(fig_nodules)
        
        # Create overlay
        original_rgb = np.stack([original]*3, axis=-1)
        if original_rgb.max() > 1:
            original_rgb = original_rgb / original_rgb.max()
        red_mask = np.zeros_like(original_rgb)
        red_mask[:,:,0] = prediction
        overlay = (1 - 0.3) * original_rgb + 0.3 * red_mask
        overlay = np.clip(overlay, 0, 1)
        
        # Overlay with boxes and centroids
        fig_overlay = plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        for i, box_info in enumerate(boxes_data):
            box = box_info["coords"]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height,
                           fill=False, color='cyan', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add centroid marker and nodule ID
            centroid_x, centroid_y = box_info["centroid"]
            plt.plot(centroid_x, centroid_y, 'ro', markersize=3)
            plt.text(x_min, y_min-5, f"#{i+1}", color='white', 
                   backgroundcolor='black', fontsize=8)
            
        plt.axis('off')
        overlay_base64 = fig_to_base64(fig_overlay)
        plt.close(fig_overlay)
        
        return {
            "visualization": {
                "original": original_base64,
                "segmentation": segmentation_base64,
                "nodules": nodules_base64,
                "overlay": overlay_base64,
                "annotations": "" # Will be handled on frontend
            },
            "nodules": nodule_data
        }
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
        raise

def create_app():
    """Create and configure the Flask app"""
    try:
        app = Flask(__name__)
        CORS(app)
        
        # Configuration
        app.config['UPLOAD_FOLDER'] = 'temp_uploads'
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Initialize model at startup
        global model
        if model is None:
            model = load_model()
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint to check if service is running"""
            return jsonify({'status': 'healthy', 'model_loaded': model is not None})
        
        @app.route('/progress', methods=['GET'])
        def get_progress():
            """Endpoint to get current processing progress"""
            global processing_progress
            return jsonify({'progress': processing_progress})
        
        @app.route('/predict_multiple', methods=['POST'])
        def predict_multiple():
            """Endpoint to process multiple images and return individual visualizations"""
            try:
                global processing_progress, model
                processing_progress = 0
                
                if model is None:
                    model = load_model()
                
                if 'files[]' not in request.files:
                    return jsonify({'error': 'No files uploaded'}), 400
                
                files = request.files.getlist('files[]')
                if not files or files[0].filename == '':
                    return jsonify({'error': 'No files selected'}), 400
                
                total_files = len(files)
                results = []
                
                # Process each file
                for index, file in enumerate(sorted(files, key=lambda x: x.filename)):
                    if not allowed_file(file.filename):
                        return jsonify({'error': f'Invalid file type for {file.filename}. Only .npy files allowed'}), 400
                    
                    try:
                        # Update progress
                        processing_progress = int((index / total_files) * 100)
                        
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
                        
                    except Exception as e:
                        print(f"Error processing file {file.filename}: {str(e)}")
                        traceback.print_exc()
                        return jsonify({'error': f'Error processing {file.filename}: {str(e)}'}), 500
                
                processing_progress = 100
                return jsonify({'results': results})
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                traceback.print_exc()
                processing_progress = 0
                return jsonify({'error': str(e)}), 500
        
        return app
    
    except Exception as e:
        print(f"Error creating Flask app: {str(e)}")
        traceback.print_exc()
        raise

# Create the Flask application
try:
    app = create_app()
except Exception as e:
    print(f"Failed to create Flask application: {str(e)}")
    traceback.print_exc()
    raise

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {str(e)}")
        traceback.print_exc()