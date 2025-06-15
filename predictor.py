"""
Prediction utilities for trained image classification models.
"""

import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


class ImagePredictor:
    """Handles predictions on new images."""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def predict_single_image(self, image_path, show_image=True):
        """Predict class for a single image."""
        # Load and preprocess image
        image = plt.imread(image_path)
        
        # Make prediction
        predictions = self.model.predict(image[None, ...], verbose=False)[0]
        predicted_class_idx = predictions.argmax()
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.3f})")
            plt.axis('off')
            plt.show()
        
        return predicted_class, confidence, predictions
    
    def predict_random_image(self, images_dir, show_image=True):
        """Predict class for a random image from the dataset."""
        image_paths = glob.glob(f"{images_dir}/*/*")
        random_image_path = random.choice(image_paths)
        
        print(f"Selected image: {random_image_path}")
        return self.predict_single_image(random_image_path, show_image)
    
    def predict_batch(self, image_paths, show_images=False):
        """Predict classes for multiple images."""
        results = []
        
        for image_path in image_paths:
            predicted_class, confidence, predictions = self.predict_single_image(
                image_path, show_image=show_images
            )
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': predictions
            })
        
        return results
    
    def predict_with_top_k(self, image_path, k=3, show_image=True):
        """Predict top-k classes for an image."""
        image = plt.imread(image_path)
        predictions = self.model.predict(image[None, ...], verbose=False)[0]
        
        # Get top-k predictions
        top_k_indices = predictions.argsort()[-k:][::-1]
        top_k_results = []
        
        for idx in top_k_indices:
            top_k_results.append({
                'class': self.class_names[idx],
                'confidence': predictions[idx]
            })
        
        if show_image:
            plt.figure(figsize=(10, 6))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis('off')
            
            # Show predictions
            plt.subplot(1, 2, 2)
            classes = [result['class'] for result in top_k_results]
            confidences = [result['confidence'] for result in top_k_results]
            
            plt.barh(classes, confidences)
            plt.xlabel('Confidence')
            plt.title(f'Top-{k} Predictions')
            plt.xlim(0, 1)
            
            plt.tight_layout()
            plt.show()
        
        return top_k_results
    
    def evaluate_predictions(self, test_images_with_labels):
        """Evaluate prediction accuracy on labeled test images."""
        correct = 0
        total = len(test_images_with_labels)
        
        for image_path, true_label in test_images_with_labels:
            predicted_class, _, _ = self.predict_single_image(image_path, show_image=False)
            if predicted_class == true_label:
                correct += 1
        
        accuracy = correct / total
        print(f"Prediction Accuracy: {accuracy:.3f} ({correct}/{total})")
        return accuracy