import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')
image_folder = 'images_resized'
print('Loading training images...')
X_train, y_train = ([], [])
for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    X_train.append(img_array)
    y_train.append(row['label_numeric'])
X_train = np.array(X_train)
y_train = np.array(y_train)
print(f'Training data shape: {X_train.shape}')
print('Loading testing images...')
X_test, y_test = ([], [])
for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row['label_numeric'])
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f'Testing data shape: {X_test.shape}')
print('\nBuilding EfficientNetB0 model...')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1), EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
print('\n--- Phase 1: Training with frozen base layers ---')
history_phase1 = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)
print('\n--- Phase 2: Fine-tuning top layers ---')
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history_phase2 = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'\nEfficientNetB0 Test Accuracy: {test_accuracy:.4f}')
model.save('glaucoma_model_efficientnet.h5')
print('EfficientNetB0 model saved as glaucoma_model_efficientnet.h5')
combined_history = {}
for key in history_phase1.history:
    combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
for key in combined_history:
    combined_history[key] = [float(v) for v in combined_history[key]]
with open('efficientnet_history.json', 'w') as f:
    json.dump(combined_history, f)
print('Training history saved as efficientnet_history.json')
