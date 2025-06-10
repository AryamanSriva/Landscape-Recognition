"""
Data loading and preprocessing utilities for image classification.
"""

import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class ImageDataLoader:
    """Handles loading and preprocessing of image datasets."""
    
    def __init__(self, images_dir, image_size=(128, 128)):
        self.images_dir = images_dir
        self.image_size = image_size
        self.class_names = None
    
    def explore_directory_structure(self):
        """Explore and print the directory structure and file counts."""
        dir_names = os.listdir(self.images_dir)
        img_paths_dict = {}
        
        for dir_name in dir_names:
            img_paths_dict[dir_name] = glob.glob(f"{self.images_dir}{dir_name}/*")
        
        print("Directory structure:")
        for dir_name, file_paths in img_paths_dict.items():
            print(f"{dir_name}: {len(file_paths)} images")
            for file_path in file_paths[:5]:
                print(f"  {file_path}")
        
        return img_paths_dict
    
    def visualize_sample_images(self, img_paths_dict, samples_per_class=5):
        """Visualize sample images from each class."""
        for dir_name, file_paths in img_paths_dict.items():
            print(f"\nSample images from {dir_name}:")
            for file_path in file_paths[:samples_per_class]:
                image = plt.imread(file_path)
                plt.figure(figsize=(4, 4))
                plt.imshow(image)
                plt.title(f"{dir_name}")
                plt.axis('off')
                plt.show()
    
    def create_dataset(self, validation_split=None, subset=None, seed=0):
        """Create a TensorFlow dataset from image directory."""
        if validation_split is None:
            dataset = keras.utils.image_dataset_from_directory(
                self.images_dir,
                image_size=self.image_size,
                label_mode="categorical"
            )
            self.class_names = dataset.class_names
            return dataset
        else:
            dataset = keras.utils.image_dataset_from_directory(
                self.images_dir,
                image_size=self.image_size,
                validation_split=validation_split,
                subset=subset,
                label_mode="categorical",
                seed=seed,
            )
            if self.class_names is None:
                self.class_names = dataset.class_names
            return dataset
    
    def create_train_validation_split(self, validation_split=0.2, seed=0):
        """Create training and validation datasets."""
        train_ds = self.create_dataset(
            validation_split=validation_split,
            subset="training",
            seed=seed
        )
        
        valid_ds = self.create_dataset(
            validation_split=validation_split,
            subset="validation",
            seed=seed
        )
        
        return train_ds, valid_ds
    
    def inspect_dataset(self, dataset):
        """Inspect dataset structure and display sample."""
        images, labels = next(iter(dataset))
        
        print(f"Images type: {type(images)}")
        print(f"Images dtype: {images.dtype}")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Display first image
        class_name = self.class_names[tf.math.argmax(labels[0])]
        print(f"First image class: {class_name}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.title(f"Sample image: {class_name}")
        plt.axis('off')
        plt.show()
        
        return images, labels