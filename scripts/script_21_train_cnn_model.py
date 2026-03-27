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
from PIL import Image
# Import core deep learning framework
import tensorflow as tf
from tensorflow.keras import layers, models
# Load the dataset from the local data directory
train_df = pd.read_csv('data/train_dataset.csv')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
X_train, y_train = ([], [])
for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    # Open the image file and convert to RGB format
    img = Image.open(img_path)
    # Normalize pixel values to standard [0, 1] range for gradient stability
    img_array = np.array(img) / 255.0
    X_train.append(img_array)
    y_train.append(row['label_numeric'])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test, y_test = ([], [])
for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    # Open the image file and convert to RGB format
    img = Image.open(img_path)
    # Normalize pixel values to standard [0, 1] range for gradient stability
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row['label_numeric'])
X_test = np.array(X_test)
y_test = np.array(y_test)
model = models.Sequential()
model.add(
    layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(
            224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
# Final binary classification head using Sigmoid for probability distribution
model.add(layers.Dense(1, activation='sigmoid'))
# Compile model using Adam optimizer and BCE loss function
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model.summary()
# Execute the model training loop
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2)
# Evaluate objective empirical loss and accuracy against test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)
model.save('glaucoma_model.h5')
print('Model saved successfully.')
