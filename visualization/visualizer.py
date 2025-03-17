"""
Visualization functions for the nodule detection application.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from utils.logging_config import logger
from utils.image_utils import fig_to_base64, get_bounding_boxes
from features.radiomic_features import extract_radiomic_features

def create_single_slice_visualization(original, prediction):
    """
    Create visualization for a single slice with layers.
    
    Args:
        original: Original image array
        prediction: Binary prediction mask
        
    Returns:
        Dictionary with visualization data and nodule information
    """
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
            
            # Log information about the nodule
            logger.info(f"Processing nodule #{i+1} with area: {box_info['area']} pixels")
            
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
                "area": int(box_info["area"]),
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
        logger.error(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise