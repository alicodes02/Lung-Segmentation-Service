import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def create_overlay(original, prediction, alpha=0.3):
    """
    Create an overlay of the prediction on the original image.
    
    Args:
        original: Original image array
        prediction: Binary prediction mask array
        alpha: Transparency of the overlay (0.0 to 1.0)
    """
    # Create RGB version of grayscale image
    original_rgb = np.stack([original]*3, axis=-1)
    
    # Normalize original image to 0-1 range if needed
    if original_rgb.max() > 1:
        original_rgb = original_rgb / original_rgb.max()
    
    # Create red mask for overlay
    red_mask = np.zeros_like(original_rgb)
    red_mask[:,:,0] = prediction  # Red channel
    
    # Combine original and mask
    overlay = (1 - alpha) * original_rgb + alpha * red_mask
    
    # Ensure values are in valid range
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def load_and_visualize(original_path, prediction_path, output_dir, save_figs=True, display_figs=True):
    """
    Load and visualize original image, prediction mask, and overlay side by side.
    """
    # Load images
    original = np.load(original_path)
    prediction = np.load(prediction_path)
    
    # Normalize original image if needed
    if original.max() > 1:
        original = original / original.max()
    
    # Create overlay
    overlay = create_overlay(original, prediction)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot prediction mask
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='hot')
    plt.title('Predicted Segmentation')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(f'Slice: {os.path.basename(original_path)}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_figs:
        # Create visualization directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        basename = os.path.basename(original_path)
        output_filename = os.path.join(output_dir, f'viz_{basename[:-4]}.png')
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f'Saved visualization to: {output_filename}')
    
    if display_figs:
        plt.show()
    else:
        plt.close()

def visualize_all_predictions(original_dir, prediction_dir, viz_output_dir):
    """
    Visualize all predictions for a patient.
    """
    # Get all original images
    original_files = sorted(glob(os.path.join(original_dir, '*_NI*.npy')))
    
    for orig_path in original_files:
        # Construct prediction path
        basename = os.path.basename(orig_path)
        pred_filename = basename.replace('_NI', '_PD')
        pred_path = os.path.join(prediction_dir, pred_filename)
        
        if os.path.exists(pred_path):
            print(f'Processing: {basename}')
            load_and_visualize(orig_path, pred_path, viz_output_dir, save_figs=True, display_figs=False)
        else:
            print(f'Warning: No prediction found for {basename}')

def main():
    # Directories
    original_dir = 'D:/LIDC-IDRI-Preprocessing/data/Image/LIDC-IDRI-0577'  # Original images directory
    prediction_dir = 'Output/'  # Directory containing predictions
    viz_output_dir = 'Visualizations/'  # Directory to save visualizations
    
    # Create visualization directory
    os.makedirs(viz_output_dir, exist_ok=True)
    
    print("Starting visualization...")
    visualize_all_predictions(original_dir, prediction_dir, viz_output_dir)
    print("Visualization complete!")
    
    # Display a single example interactively
    print("\nDisplaying first image pair interactively...")
    first_original = sorted(glob(os.path.join(original_dir, '*_NI*.npy')))[0]
    first_pred = os.path.join(prediction_dir, os.path.basename(first_original).replace('_NI', '_PD'))
    load_and_visualize(first_original, first_pred, viz_output_dir, save_figs=False, display_figs=True)

if __name__ == '__main__':
    main()