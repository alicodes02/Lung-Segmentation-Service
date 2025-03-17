"""
Configuration settings for the nodule detection application.
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
MODEL_URL = "https://www.dropbox.com/scl/fi/w446punozvpuwdkke3ea6/checkpoint.pth?rlkey=1e1y4mj4rr1fz7s94i3o6t4qf&st=nb3wohpp&dl=0"
MODEL_PATH = 'temp_model/checkpoint.pth'

# Upload folder configuration
UPLOAD_FOLDER = 'temp_uploads'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'npy'}

# Nodule detection parameters
MIN_NODULE_SIZE = 10  # Minimum size in pixels to consider a region as a nodule