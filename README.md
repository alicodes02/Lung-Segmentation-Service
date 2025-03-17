# Nodule Detection API

This application provides a Flask-based REST API for detecting nodules in medical images using a UNet model.

## Project Structure

```
nodule_detection/
├── app.py                      # Main application entry point
├── config.py                   # Configuration settings
├── utils/
│   ├── __init__.py
│   ├── logging_config.py       # Logging configuration
│   └── image_utils.py          # Image processing utilities
├── model/
│   ├── __init__.py
│   ├── loader.py               # Model loading functions
│   └── inference.py            # Model inference functions
├── features/
│   ├── __init__.py
│   ├── basic_features.py       # Basic feature calculations
│   └── radiomic_features.py    # PyRadiomics feature extraction
├── visualization/
│   ├── __init__.py
│   └── visualizer.py           # Visualization functions
└── api/
    ├── __init__.py
    ├── routes.py               # API endpoints
    └── response_utils.py       # Helper functions for API responses
```

## Key Features

- Medical image nodule detection using UNet deep learning model
- Remote model downloading capability
- Processing of multiple images in batch
- Calculation of radiomic features for detected nodules
- Visualization of detection results with bounding boxes
- Progress tracking for long-running operations

## API Endpoints

- `/health` - Check service health and model status
- `/load_model` - Explicitly load or check model loading status
- `/progress` - Get current processing progress
- `/predict_multiple` - Process multiple images and return visualizations with nodule data

## Requirements

- Python 3.6+
- PyTorch
- Flask
- SimpleITK
- PyRadiomics
- Matplotlib
- NumPy
- SciPy

## Installation

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the application: `python app.py`