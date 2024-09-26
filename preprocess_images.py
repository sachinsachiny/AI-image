# preprocess_images.py
import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def preprocess_image_directory(directory_path):
    processed_images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            processed_image = preprocess_image(image_path)
            processed_images.append(processed_image)
    return np.array(processed_images)
