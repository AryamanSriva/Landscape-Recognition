"""
Model building utilities for image classification using EfficientNet.
"""

import tensorflow as tf
from tensorflow import keras
import keras_cv


class ImageClassifierBuilder:
    """Builds and configures image classification models."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
    
    def create_efficientnet_model(self, model_preset="efficientnetv2_b0_imagenet"):
        """Create an EfficientNet-based image classifier."""
        # Load pre-trained backbone
        backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(model_preset)
        
        # Create image classifier
        self.model = keras_cv.models.ImageClassifier(
            backbone=backbone,
            num_classes=self.num_classes,
            activation="softmax",
        )
        
        print(f"Created EfficientNetV2 model with {self.num_classes} classes")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with loss function and metrics."""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_efficientnet_model() first.")
        
        loss = keras.losses.CategoricalCrossentropy()
        metric = keras.metrics.CategoricalAccuracy()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric]
        )
        
        print("Model compiled successfully")
        return self.model
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not created yet.")
        
        return self.model.summary()
    
    def create_and_compile_model(self, model_preset="efficientnetv2_b0_imagenet", learning_rate=0.001):
        """Create and compile model in one step."""
        self.create_efficientnet_model(model_preset)
        self.compile_model(learning_rate)
        return self.model