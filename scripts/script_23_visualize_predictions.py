import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('glaucoma_model.h5')
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
X_test = []
y_test = []
valid_image_names = []
for _, row in test_df.iterrows():
    image_path = os.path.join(image_folder, row['Image Name'])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        X_test.append(img_array)
        y_test.append(row['label_numeric'])
        valid_image_names.append(row['Image Name'])
    else:
        print(f'Missing image: {image_path}')
X_test = np.array(X_test)
y_test = np.array(y_test)
print('Total test images loaded:', len(X_test))
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
    img = Image.open(image_path)
    actual = y_test[i]
    predicted = predicted_labels[i][0]
    probability = predictions[i][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
    plt.show()
incorrect_indices = np.where(predicted_labels.flatten() != y_test)[0]
print('Number of incorrect predictions:', len(incorrect_indices))
print('\nDisplaying first 5 incorrect predictions...')
num_incorrect_display = min(5, len(incorrect_indices))
for idx in incorrect_indices[:num_incorrect_display]:
    image_path = os.path.join(image_folder, valid_image_names[idx])
    img = Image.open(image_path)
    actual = y_test[idx]
    predicted = predicted_labels[idx][0]
    probability = predictions[idx][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
plt.show()
print('\nDisplaying 5 random predictions...')
random_count = min(5, len(X_test))
random_indices = random.sample(range(len(X_test)), random_count)
for idx in random_indices:
    image_path = os.path.join(image_folder, valid_image_names[idx])
    img = Image.open(image_path)
    actual = y_test[idx]
    predicted = predicted_labels[idx][0]
    probability = predictions[idx][0]
    actual_text = label_map[actual]
    pred_text = label_map[predicted]
    confidence_text = interpret_probability(probability)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Actual: {actual_text} | Predicted: {pred_text}\nProb: {probability:.2f} | {confidence_text}')
    plt.axis('off')
    plt.show()
label_map = {1: 'Glaucoma', 0: 'Normal'}
actual_text = label_map[actual]
pred_text = label_map[predicted]
plt.title(f'Actual: {actual_text} | Predicted: {pred_text} | Prob: {probability:.2f}')
incorrect_indices = np.where(predicted_labels.flatten() != y_test)[0]
print('Number of incorrect predictions:', len(incorrect_indices))
