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
import json
from PIL import Image
# Import core deep learning framework
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Load the dataset from the local data directory
train_df = pd.read_csv('data/train_full_dataset.csv')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_full_dataset.csv')
image_folder = 'images_resized'
print('Loading training images and quality scores...')
X_train_images, X_train_quality, y_train = ([], [], [])
# Iterate through the metadata to load corresponding images
for _, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    # Open the image file and convert to RGB format
    img = Image.open(img_path).convert('RGB')
    # Resize standard dimensions suitable for ImageNet pre-trained backbones
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    X_train_images.append(img_array)
    X_train_quality.append(row['quality_normalized'])
    y_train.append(row['label_numeric'])
X_train_images = np.array(X_train_images)
X_train_quality = np.array(X_train_quality, dtype=np.float32)
y_train = np.array(y_train)
print(f'Training images shape: {X_train_images.shape}')
print(f'Training quality scores shape: {X_train_quality.shape}')
print('Loading testing images and quality scores...')
X_test_images, X_test_quality, y_test = ([], [], [])
# Iterate through the metadata to load corresponding images
for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    # Open the image file and convert to RGB format
    img = Image.open(img_path).convert('RGB')
    # Resize standard dimensions suitable for ImageNet pre-trained backbones
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    X_test_images.append(img_array)
    X_test_quality.append(row['quality_normalized'])
    y_test.append(row['label_numeric'])
X_test_images = np.array(X_test_images)
X_test_quality = np.array(X_test_quality, dtype=np.float32)
y_test = np.array(y_test)
print(f'Testing images shape: {X_test_images.shape}')
print(f'Testing quality scores shape: {X_test_quality.shape}')
print('\nBuilding Hybrid ResNet50 model (Quality-as-Feature)...')
# Define the input layer tensor
image_input = tf.keras.Input(shape=(224, 224, 3), name='image_input')
# Load the ResNet50 base architecture
resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=image_input)
resnet_base.trainable = False
x = layers.GlobalAveragePooling2D()(resnet_base.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
# Define the input layer tensor
quality_input = tf.keras.Input(shape=(1,), name='quality_input')
y = layers.Dense(16, activation='relu')(quality_input)
# Critical step: Mathematically fuse the visual features with the
# numerical Quality Score
combined = layers.Concatenate()([x, y])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dropout(0.3)(z)
# Final binary classification head using Sigmoid for probability distribution
final_output = layers.Dense(1, activation='sigmoid', name='glaucoma_output')(z)
hybrid_model = models.Model(
    inputs=[
        image_input,
        quality_input],
    outputs=final_output)
# Compile model using Adam optimizer and BCE loss function
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])
hybrid_model.summary()
# Define callbacks: gracefully decay learning rate if the model stops improving
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1)]
print('\n--- Phase 1: Training with frozen base layers ---')
# Execute the model training loop
history_phase1 = hybrid_model.fit(x=[X_train_images,
                                     X_train_quality],
                                  y=y_train,
                                  epochs=10,
                                  batch_size=32,
                                  validation_split=0.2,
                                  callbacks=callbacks)
print('\n--- Phase 2: Fine-tuning top layers ---')
resnet_base.trainable = True
for layer in resnet_base.layers[:-20]:
    layer.trainable = False
# Compile model using Adam optimizer and BCE loss function
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy'])
# Execute the model training loop
history_phase2 = hybrid_model.fit(x=[X_train_images,
                                     X_train_quality],
                                  y=y_train,
                                  epochs=10,
                                  batch_size=32,
                                  validation_split=0.2,
                                  callbacks=callbacks)
# Evaluate objective empirical loss and accuracy against test set
test_loss, test_accuracy = hybrid_model.evaluate(
    [X_test_images, X_test_quality], y_test)
print(f'\nHybrid ResNet50 Test Accuracy: {test_accuracy:.4f}')
hybrid_model.save('hybrid_resnet50.h5')
print('Hybrid ResNet50 model saved as hybrid_resnet50.h5')
combined_history = {}
for key in history_phase1.history:
    combined_history[key] = history_phase1.history[key] + \
        history_phase2.history[key]
for key in combined_history:
    combined_history[key] = [float(v) for v in combined_history[key]]
with open('results/hybrid_resnet50_history.json', 'w') as f:
    json.dump(combined_history, f)
print('Training history saved as hybrid_resnet50_history.json')
