import os
import numpy as np
import torch
import torch.nn as nn
from Unet.unet_model import UNet  # Assuming you have the same model structure
import glob

def get_device():
    """Determine the available device (CPU or CUDA)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_model(checkpoint_path, device, model_type='UNET'):
    """Load the trained model from checkpoint."""
    if model_type == 'UNET':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    # Load checkpoint
    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        # Load checkpoint to CPU
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Handle both DataParallel and regular model state dictionaries
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        # If model was trained with DataParallel but running on CPU
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model

def process_single_image(image_path, model, device):
    """Process a single .npy file through the model."""
    # Load and preprocess the image
    image = np.load(image_path)
    # Add batch and channel dimensions
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()
        output = np.squeeze(output)  # Remove batch and channel dimensions
    
    return output

def process_patient_images(patient_dir, output_dir, model, device, file_pattern='*_NI*.npy'):  # Changed pattern here
    """Process all .npy files for a patient and save predictions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all matching .npy files in the patient directory
    image_paths = glob.glob(os.path.join(patient_dir, file_pattern))
    print(f"Found {len(image_paths)} images to process")
    print(f"Looking for files in: {patient_dir}")
    print(f"Using pattern: {file_pattern}")
    
    if len(image_paths) == 0:
        print("Available files in directory:")
        for file in os.listdir(patient_dir):
            if file.endswith('.npy'):
                print(file)
        return
    
    for i, image_path in enumerate(image_paths, 1):
        # Get the filename and create output path
        filename = os.path.basename(image_path)
        output_filename = filename.replace('_NI', '_PD')  # Changed replacement pattern
        output_path = os.path.join(output_dir, output_filename)
        
        # Process the image
        prediction = process_single_image(image_path, model, device)
        
        # Save the prediction
        np.save(output_path, prediction)
        print(f"Processed and saved: {output_path} ({i}/{len(image_paths)})")

def main():
    checkpoint_path = 'model_outputs/UNET_base/checkpoint.pth'
    patient_dir = 'D:/LIDC-IDRI-Preprocessing/data/Image/LIDC-IDRI-0577'
    output_dir = 'Output/'
    
    # Verify directory exists
    if not os.path.exists(patient_dir):
        print(f"Error: Directory not found: {patient_dir}")
        return
        
    # Get device
    device = get_device()
    
    # Load model
    print("Loading model...")
    try:
        model = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Process all images for the patient
    print("Processing patient images...")
    process_patient_images(patient_dir, output_dir, model, device)
    
    print("Processing complete!")

if __name__ == '__main__':
    main()

