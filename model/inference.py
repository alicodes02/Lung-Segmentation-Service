"""
Model inference functions for the nodule detection application.
"""
import torch
import numpy as np
from utils.logging_config import logger
from config import DEVICE

def process_image(image_array, model):
    """
    Process a single image through the model.
    
    Args:
        image_array: NumPy array of the image
        model: UNet model for inference
        
    Returns:
        Binary mask of predicted nodules
    """
    try:
        image = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        image = image.to(DEVICE)
        
        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output)
        
        return output
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise