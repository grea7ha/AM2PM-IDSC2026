import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
train_df = pd.read_csv('data/train_dataset.csv')
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
X_train, y_train = ([], [])
for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0
    X_train.append(img_array)
    y_train.append(row['label_numeric'])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test, y_test = ([], [])
for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row['label_numeric'])
X_test = np.array(X_test)
y_test = np.array(y_test)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)
model.save('glaucoma_model.h5')
print('Model saved successfully.')
