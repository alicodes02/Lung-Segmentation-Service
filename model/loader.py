"""
Model loading functions for the nodule detection application.
"""
import os
import torch
import requests
from Unet.unet_model import UNet
from utils.logging_config import logger
from config import DEVICE, MODEL_URL, MODEL_PATH

def download_model_from_url(url, destination_path):
    """
    Download the model file from a URL.
    
    Args:
        url: URL to download from
        destination_path: Path to save the downloaded model
        
    Returns:
        Path to the downloaded model or None if download failed
    """
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Special handling for Dropbox URLs
        if 'dropbox.com' in url:
            # Convert to direct download link if it's not already
            if 'dl=0' in url:
                url = url.replace('dl=0', 'dl=1')
            elif 'dl=' not in url:
                if '?' in url:
                    url += '&dl=1'
                else:
                    url += '?dl=1'
            logger.info(f"Modified Dropbox URL for direct download: {url}")
        
        # Download the file
        logger.info(f"Downloading model from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Verify content type is not HTML
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            # This might be a webpage instead of a file
            logger.warning(f"Server returned HTML content instead of a file (Content-Type: {content_type})")
            logger.debug("First 100 bytes of response: %s", response.content[:100])
            # Try to check if this is a redirect page with a download button
            if b'<!DOCTYPE html>' in response.content[:20]:
                raise Exception(f"Received HTML content instead of model file. Please ensure the URL provides direct file download.")
        
        # Get total file size for progress reporting
        total_size = int(response.headers.get('content-length', 0))
        
        # Save the file
        with open(destination_path, 'wb') as f:
            if total_size == 0:  # No content length header
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = int(100 * downloaded / total_size)
                        if percent % 10 == 0:
                            logger.info(f"Download progress: {percent}%")
        
        # Verify the downloaded file is a valid PyTorch model
        with open(destination_path, 'rb') as f:
            first_bytes = f.read(10)
            if b'<!DOCTYPE' in first_bytes or b'<html' in first_bytes:
                os.remove(destination_path)
                raise Exception("Downloaded file appears to be HTML, not a PyTorch model file")
        
        logger.info(f"Model file downloaded to {destination_path}")
        return destination_path
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_model():
    """
    Initialize the model from remote URL or local cache.
    
    Returns:
        Loaded PyTorch model ready for inference
    """
    try:
        # Check if model already exists locally
        if os.path.exists(MODEL_PATH):
            logger.info(f"Model file already exists at {MODEL_PATH}, skipping download")
            model_path = MODEL_PATH
        else:
            logger.info("Model not found locally. Downloading from remote server...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Download the model file from URL
            model_path = download_model_from_url(MODEL_URL, MODEL_PATH)
            
            if not model_path or not os.path.exists(model_path):
                raise Exception("Failed to download model file from URL")
            
            logger.info(f"Model downloaded successfully to {model_path}")
        
        # Initialize the model
        logger.info("Loading model into memory...")
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel state dict if necessary
        if 'module' in list(state_dict.keys())[0]:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise