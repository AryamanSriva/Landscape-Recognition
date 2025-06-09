"""
Configuration file for image classification project.
"""

import os

# Data Configuration
IMAGES_DIR = "/usercode/data/images/"  # Path to your image dataset
IMAGE_SIZE = (128, 128)  # Input image size for the model
VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation
SEED = 0  # Random seed for reproducibility

# Model Configuration
NUM_CLASSES = 3  # Number of classes in your dataset
MODEL_PRESET = "efficientnetv2_b0_imagenet"  # EfficientNet model variant
LEARNING_RATE = 0.001

# Training Configuration
EPOCHS = 10
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5

# Paths
MODEL_SAVE_PATH = "models/trained_model.keras"
LOGS_DIR = "logs/"
RESULTS_DIR = "results/"

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Visualization Configuration
SAMPLES_PER_CLASS = 5
FIGURE_SIZE = (10, 8)
DPI = 100