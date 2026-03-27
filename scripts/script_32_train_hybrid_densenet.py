"""
Script 32: Train Hybrid DenseNet121 (Quality-Aware Model)
==========================================================
IDSC 2026 | Team AM2PM | Quality-Aware Glaucoma Triage

This script implements the core innovation of our submission: a dual-input
hybrid neural network that processes both the retinal image AND its quality
score simultaneously.

Architecture:
    Branch A (Visual):  DenseNet121 (ImageNet pre-trained) → GlobalAvgPool
                        → BatchNorm → Dense(256) → Dropout(0.5)
    Branch B (Quality): Dense(16, ReLU) on the normalised Quality Score
    Merge:              Concatenate → Dense(64) → Dropout(0.3) → Sigmoid

Training Strategy (Two-Phase Transfer Learning):
    Phase 1: Freeze DenseNet121 base; train top layers for 10 epochs (lr=1e-3)
    Phase 2: Unfreeze top 20 base layers; fine-tune for 10 epochs (lr=1e-4)

Output:
    - hybrid_densenet.h5           : Saved model weights
    - hybrid_densenet_history.json : Full training history (20 epochs)
"""

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_FOLDER  = 'images_resized'
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-4

# ── Load dataset metadata ─────────────────────────────────────────────────────
train_df = pd.read_csv('data/train_full_dataset.csv')
test_df  = pd.read_csv('data/test_full_dataset.csv')


def load_images_and_quality(dataframe, image_folder):
    """
    Load all retinal images and their associated quality scores from disk.

    Args:
        dataframe    : DataFrame with 'Image Name', 'quality_normalized',
                       and 'label_numeric' columns.
        image_folder : Path to the folder containing resized images.

    Returns:
        Tuple of (images array, quality scores array, labels array)
    """
    images  = []
    quality = []
    labels  = []

    for _, row in dataframe.iterrows():
        img_path = os.path.join(image_folder, row['Image Name'])

        # Load image, ensure RGB (3 channels), and normalise to [0, 1]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0

        images.append(img_array)
        quality.append(row['quality_normalized'])
        labels.append(row['label_numeric'])

    return (
        np.array(images),
        np.array(quality, dtype=np.float32),
        np.array(labels)
    )


# ── Load training and testing data ────────────────────────────────────────────
print('Loading training images and quality scores...')
X_train_images, X_train_quality, y_train = load_images_and_quality(train_df, IMAGE_FOLDER)
print(f'  Training images shape   : {X_train_images.shape}')
print(f'  Training quality shape  : {X_train_quality.shape}')

print('\nLoading testing images and quality scores...')
X_test_images, X_test_quality, y_test = load_images_and_quality(test_df, IMAGE_FOLDER)
print(f'  Testing images shape    : {X_test_images.shape}')
print(f'  Testing quality shape   : {X_test_quality.shape}')

# ── Build the Hybrid Model ────────────────────────────────────────────────────
print('\nBuilding Hybrid DenseNet121 (Quality-as-Feature) model...')

# Branch A: Image input → DenseNet121 feature extractor
image_input   = tf.keras.Input(shape=(224, 224, 3), name='image_input')
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
densenet_base.trainable = False  # Frozen during Phase 1

x = layers.GlobalAveragePooling2D()(densenet_base.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Branch B: Quality score input → small dense projection
quality_input = tf.keras.Input(shape=(1,), name='quality_input')
q = layers.Dense(16, activation='relu')(quality_input)

# Merge: Concatenate image features with quality features
combined = layers.Concatenate()([x, q])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dropout(0.3)(z)

# Output: Binary classification (Glaucoma vs Normal)
final_output = layers.Dense(1, activation='sigmoid', name='glaucoma_output')(z)

hybrid_model = models.Model(
    inputs=[image_input, quality_input],
    outputs=final_output
)

hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
hybrid_model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # Halve the learning rate if validation loss stalls for 3 epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    # Stop early and restore best weights if validation loss stalls for 5 epochs
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

# ── Phase 1: Feature Extraction (frozen base) ─────────────────────────────────
print('\n--- Phase 1: Feature Extraction (DenseNet121 base frozen) ---')
history_phase1 = hybrid_model.fit(
    x=[X_train_images, X_train_quality],
    y=y_train,
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks
)

# ── Phase 2: Fine-Tuning (top 20 base layers unfrozen) ───────────────────────
print('\n--- Phase 2: Fine-Tuning (top 20 DenseNet121 layers unfrozen) ---')
densenet_base.trainable = True

# Keep lower layers frozen to preserve lower-level ImageNet features
for layer in densenet_base.layers[:-20]:
    layer.trainable = False

# Recompile with a much lower learning rate to avoid overwriting pre-trained weights
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_phase2 = hybrid_model.fit(
    x=[X_train_images, X_train_quality],
    y=y_train,
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks
)

# ── Evaluate on held-out test set ─────────────────────────────────────────────
test_loss, test_accuracy = hybrid_model.evaluate(
    [X_test_images, X_test_quality], y_test
)
print(f'\nHybrid DenseNet121 — Test Accuracy: {test_accuracy:.4f}')

# ── Save model ────────────────────────────────────────────────────────────────
hybrid_model.save('hybrid_densenet.h5')
print('Model saved: hybrid_densenet.h5')

# ── Save training history ─────────────────────────────────────────────────────
# Combine both phases into a single continuous history for plotting
combined_history = {}
for key in history_phase1.history:
    combined_history[key] = (
        history_phase1.history[key] + history_phase2.history[key]
    )

# Convert numpy floats to native Python floats for JSON serialisation
for key in combined_history:
    combined_history[key] = [float(v) for v in combined_history[key]]

with open('results/hybrid_densenet_history.json', 'w') as f:
    json.dump(combined_history, f, indent=2)

print('Training history saved: hybrid_densenet_history.json')
