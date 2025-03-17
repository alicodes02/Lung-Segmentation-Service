"""
Image processing utilities for the nodule detection application.
"""
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.patches import Rectangle
from utils.logging_config import logger
from config import MIN_NODULE_SIZE

matplotlib.use('Agg')  # Use non-interactive backend

def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string of the figure
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error in fig_to_base64: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_bounding_boxes(prediction, min_size=MIN_NODULE_SIZE):
    """
    Get bounding boxes around nodules in a binary prediction mask.
    
    Args:
        prediction: Binary prediction mask
        min_size: Minimum size in pixels to consider a region as a nodule
        
    Returns:
        List of dictionaries containing bounding box information
    """
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
                "mask": nodule_mask,
                "area": np.sum(nodule_mask)  # Store area for reference
            })
        
        return boxes
    except Exception as e:
        logger.error(f"Error in get_bounding_boxes: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def allowed_file(filename):
    """
    Check if file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Boolean indicating if the file extension is allowed
    """
    from config import ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS