"""
Main script for image classification using EfficientNet and Keras.
"""

import os
from data_loader import ImageDataLoader
from model_builder import ImageClassifierBuilder
from trainer import ModelTrainer
from predictor import ImagePredictor


def main():
    """Main function to run the complete image classification pipeline."""
    
    # Configuration
    IMAGES_DIR = "/usercode/data/images/"  # Update this path as needed
    IMAGE_SIZE = (128, 128)
    VALIDATION_SPLIT = 0.2
    EPOCHS = 2
    NUM_CLASSES = 3  # Update based on your dataset
    
    print("Starting Image Classification Pipeline...")
    print("=" * 50)
    
    # Step 1: Initialize data loader
    print("Step 1: Initializing data loader...")
    data_loader = ImageDataLoader(IMAGES_DIR, IMAGE_SIZE)
    
    # Step 2: Explore directory structure
    print("\nStep 2: Exploring directory structure...")
    img_paths_dict = data_loader.explore_directory_structure()
    
    # Step 3: Visualize sample images (optional - comment out for faster execution)
    print("\nStep 3: Visualizing sample images...")
    # data_loader.visualize_sample_images(img_paths_dict, samples_per_class=2)
    
    # Step 4: Create datasets
    print("\nStep 4: Creating datasets...")
    train_ds, valid_ds = data_loader.create_train_validation_split(
        validation_split=VALIDATION_SPLIT
    )
    
    # Step 5: Inspect dataset
    print("\nStep 5: Inspecting dataset...")
    data_loader.inspect_dataset(train_ds)
    
    # Step 6: Build model
    print("\nStep 6: Building model...")
    model_builder = ImageClassifierBuilder(NUM_CLASSES)
    model = model_builder.create_and_compile_model()
    
    # Step 7: Train model
    print("\nStep 7: Training model...")
    trainer = ModelTrainer(model)
    history = trainer.train(train_ds, valid_ds, epochs=EPOCHS)
    
    # Step 8: Evaluate model
    print("\nStep 8: Evaluating model...")
    loss, accuracy = trainer.evaluate(valid_ds)
    
    # Step 9: Plot training history
    print("\nStep 9: Plotting training history...")
    trainer.plot_training_history()
    
    # Step 10: Make predictions
    print("\nStep 10: Making predictions on new images...")
    predictor = ImagePredictor(model, data_loader.class_names)
    
    # Predict on a random image
    predicted_class, confidence, _ = predictor.predict_random_image(IMAGES_DIR)
    print(f"Random prediction: {predicted_class} (Confidence: {confidence:.3f})")
    
    # Save model (optional)
    model_save_path = "trained_model.keras"
    trainer.save_model(model_save_path)
    
    print("\n" + "=" * 50)
    print("Image Classification Pipeline Completed!")
    print(f"Final Model Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # Check if TensorFlow can detect GPU
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    print()
    
    main()