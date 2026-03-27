"""
Quality-Aware Glaucoma Triage System Pipeline
=========================================================
Author: Thanush Govindarajoo, Gunasree R
Institution: National University of Malaysia, Anna University, India
Dataset: Hillel Yaffe Glaucoma Dataset (HYGD)

Project: Mathematics for Hope in Healthcare (IDSC 2026)
Description:
This script is part of our step-by-step modular pipeline for detecting
Glaucomatous Optic Neuropathy. We explicitly modularized our code so
each mathematical step (data loading, splitting, training, evaluation)
can be verified independently.
"""

# Import necessary data structuring libraries
import pandas as pd
# Import numerical computing library
import numpy as np
import os
import random
from PIL import Image
# Import core deep learning framework
import tensorflow as tf
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('glaucoma_model.h5')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
X_test = []
y_test = []
valid_image_names = []
# Iterate through the metadata to load corresponding images
for _, row in test_df.iterrows():
    image_path = os.path.join(image_folder, row['Image Name'])
    if os.path.exists(image_path):
        # Open the image file and convert to RGB format
        img = Image.open(image_path).convert('RGB')
        # Resize standard dimensions suitable for ImageNet pre-trained
        # backbones
        img = img.resize((224, 224))
        # Normalize pixel values to standard [0, 1] range for gradient
        # stability
        img_array = np.array(img) / 255.0
        X_test.append(img_array)
        y_test.append(row['label_numeric'])
        valid_image_names.append(row['Image Name'])
    else:
        print(f'Missing image: {image_path}')
X_test = np.array(X_test)
y_test = np.array(y_test)
print('Total test images loaded:', len(X_test))
# Generate probabilistic predictions on the unseen test dataset
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
label_map = {1: 'Glaucoma', 0: 'Normal'}


def interpret_probability(prob):
    if prob >= 0.95:
        return 'Very confident glaucoma'
    elif prob >= 0.7:
        return 'Moderate confidence'
    elif prob >= 0.52:
        return 'Uncertain prediction'
    else:
        return 'Low probability of glaucoma'


print('\nDisplaying first 10 predictions...')
num_display = min(10, len(X_test))
for i in range(num_display):
    image_path = os.path.join(image_folder, valid_image_names[i])
    # Open the image file and convert to RGB format
    img = Image.open(image_path)
    actual = y_test[i]
    predicted = predicted_labels[i][0]
    probability = predictions[i][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(
        f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
    plt.savefig(f'results/prediction_{i}.png')
    # plt.show()
incorrect_indices = np.where(predicted_labels.flatten() != y_test)[0]
print('Number of incorrect predictions:', len(incorrect_indices))
print('\nDisplaying first 5 incorrect predictions...')
num_incorrect_display = min(5, len(incorrect_indices))
for idx in incorrect_indices[:num_incorrect_display]:
    image_path = os.path.join(image_folder, valid_image_names[idx])
    # Open the image file and convert to RGB format
    img = Image.open(image_path)
    actual = y_test[idx]
    predicted = predicted_labels[idx][0]
    probability = predictions[idx][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(
        f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
plt.savefig('results/incorrect_predictions.png')
# plt.show()
print('\nDisplaying 5 random predictions...')
random_count = min(5, len(X_test))
random_indices = random.sample(range(len(X_test)), random_count)
for idx in random_indices:
    image_path = os.path.join(image_folder, valid_image_names[idx])
    # Open the image file and convert to RGB format
    img = Image.open(image_path)
    actual = y_test[idx]
    predicted = predicted_labels[idx][0]
    probability = predictions[idx][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(
        f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
    plt.savefig(f'results/random_prediction_{idx}.png')
    # plt.show()
label_map = {1: 'Glaucoma', 0: 'Normal'}
actual_text = label_map[actual]
pred_text = label_map[predicted]
plt.title(
    f'Actual: {actual_text} | Predicted: {pred_text} | Prob: {probability:.2f}')
incorrect_indices = np.where(predicted_labels.flatten() != y_test)[0]
print('Number of incorrect predictions:', len(incorrect_indices))
