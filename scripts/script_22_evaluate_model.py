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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
model = tf.keras.models.load_model('glaucoma_model.h5')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
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
# Generate probabilistic predictions on the unseen test dataset
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, predicted_labels)
print('Confusion Matrix:')
print(cm)
print('\nClassification Report:')
print(classification_report(y_test, predicted_labels))
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
        'Normal', 'Glaucoma'], yticklabels=[
            'Normal', 'Glaucoma'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
# plt.show()
test_df['prediction'] = predicted_labels
test_df['probability'] = predictions
test_df.to_csv('data/test_predictions.csv', index=False)
print('Predictions saved to test_predictions.csv')
