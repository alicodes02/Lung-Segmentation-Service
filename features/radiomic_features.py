"""
Radiomic feature extraction functions for the nodule detection application.
"""
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from utils.logging_config import logger
from features.basic_features import calculate_basic_features

def init_radiomics_extractor():
    """
    Initialize the radiomics feature extractor with default settings.
    
    Returns:
        Radiomics feature extractor object
    """
    try:
        # Create a simplified extractor - avoid the problematic shape features entirely
        settings = {
            'force2D': True,  # Ensure 2D mode is active
            'force2Ddimension': 0,  # Slice direction
            'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'verbose': False  # Reduce logging noise
        }
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Only enable specific feature classes that are known to work
        extractor.disableAllFeatures()
        
        # Only enable first-order features - most reliable
        extractor.enableFeaturesByName(firstorder=True)
        
        # Explicitly disable shape features to avoid errors
        try:
            extractor.enableFeaturesByName(shape=False)
        except:
            logger.info("Shape features already disabled")
        
        logger.info("Radiomics extractor initialized with firstorder features only")
        return extractor
    except Exception as e:
        logger.error(f"Error initializing radiomics extractor: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_radiomic_features(image, mask):
    """
    Extract features using both PyRadiomics and custom calculations.
    
    Args:
        image: Image array
        mask: Binary mask of the nodule
        
    Returns:
        Dictionary of feature name and value pairs
    """
    try:
        # Always calculate our own basic features first (reliable fallback)
        basic_features = calculate_basic_features(image, mask)
        
        # Skip PyRadiomics for very small nodules
        if np.sum(mask) < 30:
            logger.info(f"Nodule too small ({np.sum(mask)} pixels) for reliable PyRadiomics analysis. Using basic features only.")
            return basic_features
        
        # Try to add additional PyRadiomics features if possible
        try:
            # Force 3D format - PyRadiomics works with 3D images
            if image.ndim == 2:
                image_3d = np.expand_dims(image, axis=0)
            else:
                image_3d = image
                
            if mask.ndim == 2:
                mask_3d = np.expand_dims(mask, axis=0)
            else:
                mask_3d = mask
            
            # Important: Make sure mask only has 0 and 1 values (no other labels)
            mask_3d = (mask_3d > 0).astype(np.int32)
                
            # Convert numpy arrays to SimpleITK images with proper metadata
            image_sitk = sitk.GetImageFromArray(image_3d.astype(np.float32))
            mask_sitk = sitk.GetImageFromArray(mask_3d)
            
            # Set spacing and direction to ensure proper feature calculation
            spacing = [1.0, 1.0, 1.0]  # Default spacing of 1mm
            image_sitk.SetSpacing(spacing)
            mask_sitk.SetSpacing(spacing)
            
            # Initialize extractor
            extractor = init_radiomics_extractor()
            if extractor is None:
                logger.warning("Could not initialize PyRadiomics. Using basic features only.")
                return basic_features
                
            # Extract features
            radiomics_features = extractor.execute(image_sitk, mask_sitk)
            
            # Map PyRadiomics features to our standard feature names
            feature_map = {
                'firstorder_Mean': 'Mean Intensity',
                'firstorder_Energy': 'Energy',
                'firstorder_Entropy': 'Entropy',
                'firstorder_Kurtosis': 'Kurtosis',
                'firstorder_Skewness': 'Skewness',
                'firstorder_TotalEnergy': 'Total Energy',
                'firstorder_Uniformity': 'Uniformity'
            }
            
            # Add any found PyRadiomics features to our basic features
            for pyrad_name, display_name in feature_map.items():
                if pyrad_name in radiomics_features:
                    basic_features[display_name] = float(radiomics_features[pyrad_name])
            
            logger.info(f"Successfully enhanced basic features with PyRadiomics firstorder features")
            
        except Exception as e:
            logger.warning(f"PyRadiomics extraction failed: {str(e)}, using basic features only")
        
        return basic_features
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        # Ensure we always return something
        return {"Area (pixels)": float(np.sum(mask))}