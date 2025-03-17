"""
Basic feature calculation functions for the nodule detection application.
"""
import numpy as np
from scipy import ndimage
from utils.logging_config import logger

def calculate_basic_features(image, mask):
    """
    Calculate basic features without using PyRadiomics or scikit-image.
    
    Args:
        image: Image array
        mask: Binary mask of the nodule
        
    Returns:
        Dictionary of feature name and value pairs
    """
    try:
        # Count pixels in the mask
        area = np.sum(mask)
        if area == 0:
            return {"Area (pixels)": 0.0}
        
        # Get masked image
        masked_image = image * mask
        nonzero_values = masked_image[mask > 0]
        
        # Basic intensity features
        mean = np.mean(nonzero_values)
        variance = np.var(nonzero_values)
        min_val = np.min(nonzero_values)
        max_val = np.max(nonzero_values)
        
        # Calculate perimeter (approximate using contour)
        # Erode the mask to find inner pixels
        eroded = ndimage.binary_erosion(mask)
        # Perimeter pixels = original - eroded
        perimeter_pixels = mask.astype(int) - eroded.astype(int)
        perimeter = np.sum(perimeter_pixels)
        
        # Find bounding box and calculate elongation
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 0)
        cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 0)
        
        width = cmax - cmin + 1
        height = rmax - rmin + 1
        
        # Calculate center of mass (centroid)
        labeled_mask, num = ndimage.label(mask)
        if num > 0:
            cy, cx = ndimage.center_of_mass(mask, labeled_mask, 1)
        else:
            cy, cx = height/2, width/2
            
        elongation = float(max(width, height) / max(1, min(width, height)))
        
        # Circularity / Sphericity approximation
        # Perfect circle has 4*pi*area / perimeter² = 1
        perimeter_squared = max(1, perimeter * perimeter)
        circularity = (4 * np.pi * area) / perimeter_squared
        
        # Calculate entropy
        try:
            histogram, _ = np.histogram(nonzero_values, bins=10)
            histogram = histogram / np.sum(histogram)
            entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        except:
            entropy = 0.0
        
        # Calculate standard deviation
        std_dev = np.std(nonzero_values)
        
        # Calculate simple measures of texture
        # Rough approximation of contrast
        contrast = (max_val - min_val) / (max_val + min_val + 1e-10)
        
        # Simple "spread" metric
        spread = std_dev / (mean + 1e-10)
        
        # Create feature dictionary
        features = {
            "Area (pixels)": float(area),
            "Perimeter": float(perimeter),
            "Elongation": elongation,
            "Sphericity": float(circularity),
            "Mean Intensity": float(mean),
            "Maximum": float(max_val),
            "Minimum": float(min_val),
            "Range": float(max_val - min_val),
            "Variance": float(variance),
            "Standard Deviation": float(std_dev),
            "Entropy": float(entropy),
            "Width": float(width),
            "Height": float(height),
            "Centroid X": float(cx),
            "Centroid Y": float(cy),
            "Contrast": float(contrast),
            "Texture Spread": float(spread)
        }
        
        # Calculate simple texture metrics without using scikit-image
        try:
            # A simplified homogeneity metric
            values_normalized = (nonzero_values - min_val) / (max_val - min_val + 1e-10)
            homogeneity = 1.0 - np.std(values_normalized)
            features["Homogeneity"] = float(homogeneity)
            
            # Energy (sum of squared values)
            energy = np.sum(values_normalized ** 2) / (area + 1e-10)
            features["Energy"] = float(energy)
            
            # Simplified uniformity
            hist, _ = np.histogram(values_normalized, bins=10)
            hist = hist / np.sum(hist)
            uniformity = np.sum(hist ** 2)
            features["Uniformity"] = float(uniformity)
            
        except Exception as e:
            logger.warning(f"Could not calculate simple texture features: {str(e)}")
            
        return features
    except Exception as e:
        logger.error(f"Error calculating basic features: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"Area (pixels)": float(np.sum(mask))}