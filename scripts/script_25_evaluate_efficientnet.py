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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import numerical computing library
import numpy as np
import os
import json
from PIL import Image
# Import core deep learning framework
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib
matplotlib.use('Agg')
print('Loading EfficientNetB0 model...')
model = tf.keras.models.load_model('glaucoma_model_efficientnet.h5')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_dataset.csv')
image_folder = 'images_resized'
X_test, y_test = ([], [])
valid_image_names = []
# Iterate through the metadata to load corresponding images
for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    if os.path.exists(img_path):
        # Open the image file and convert to RGB format
        img = Image.open(img_path).convert('RGB')
        # Resize standard dimensions suitable for ImageNet pre-trained
        # backbones
        img = img.resize((224, 224))
        # Normalize pixel values to standard [0, 1] range for gradient
        # stability
        img_array = np.array(img) / 255.0
        X_test.append(img_array)
        y_test.append(row['label_numeric'])
        valid_image_names.append(row['Image Name'])
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f'Test images loaded: {len(X_test)}')
# Generate probabilistic predictions on the unseen test dataset
predictions_prob = model.predict(X_test).flatten()
predicted_labels = (predictions_prob > 0.5).astype(int)
metrics = {
    'accuracy': float(
        accuracy_score(
            y_test, predicted_labels)), 'precision': float(
                precision_score(
                    y_test, predicted_labels, zero_division=0)), 'recall': float(
                        recall_score(
                            y_test, predicted_labels, zero_division=0)), 'f1_score': float(
                                f1_score(
                                    y_test, predicted_labels, zero_division=0)), 'roc_auc': float(
                                        roc_auc_score(
                                            y_test, predictions_prob))}
print('\n===== EfficientNetB0 Evaluation Metrics =====')
for metric_name, value in metrics.items():
    print(f'  {metric_name:>12s}: {value:.4f}')
print('\nClassification Report:')
print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=[
            'Normal',
            'Glaucoma']))
with open('results/efficientnet_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics saved to efficientnet_metrics.json')
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
        'Normal', 'Glaucoma'], yticklabels=[
            'Normal', 'Glaucoma'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('EfficientNetB0 - Confusion Matrix')
plt.tight_layout()
# Save the generated plot to disk rather than displaying interactively
plt.savefig('efficientnet_cm.png', dpi=150)
plt.close()
print('Confusion matrix saved to efficientnet_cm.png')
# Calculate ROC curve coordinates for visual performance evaluation
fpr, tpr, _ = roc_curve(y_test, predictions_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"EfficientNetB0 (AUC = {metrics['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('EfficientNetB0 - ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
# Save the generated plot to disk rather than displaying interactively
plt.savefig('efficientnet_roc.png', dpi=150)
plt.close()
print('ROC curve saved to efficientnet_roc.png')
print('\nGenerating Grad-CAM heatmaps...')


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """Generate Grad-CAM heatmap for a given image."""
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sub_layer in reversed(layer.layers):
                        if hasattr(sub_layer, 'output') and len(
                                sub_layer.output.shape) == 4:
                            last_conv_layer_name = sub_layer.name
                            grad_model = tf.keras.Model(
                                inputs=model.input, outputs=[
                                    layer.get_layer(last_conv_layer_name).output, model.output])
                            break
                    if last_conv_layer_name:
                        break
    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output])
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    conv_output = layer.get_layer(last_conv_layer_name).output
                    grad_model = tf.keras.Model(
                        inputs=model.input, outputs=[
                            conv_output, model.output])
                    break
                except ValueError:
                    continue
    img_tensor = tf.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Compute the Grad-CAM heatmap highlighting visual clinical markers
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    # Compute the Grad-CAM heatmap highlighting visual clinical markers
    heatmap = tf.squeeze(heatmap)
    # Compute the Grad-CAM heatmap highlighting visual clinical markers
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-08)
    return heatmap.numpy()


def overlay_gradcam(img_array, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img = np.uint8(255 * img_array)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(
        Image.fromarray(heatmap_resized).resize(
            (img.shape[1], img.shape[0])))
    colormap = plt.cm.jet
    heatmap_colored = colormap(heatmap_resized / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    superimposed = np.uint8(heatmap_colored * alpha + img * (1 - alpha))
    return superimposed


gradcam_dir = 'gradcam_efficientnet'
os.makedirs(gradcam_dir, exist_ok=True)
label_map = {1: 'Glaucoma', 0: 'Normal'}
num_samples = min(5, len(X_test))
for i in range(num_samples):
    # Compute the Grad-CAM heatmap highlighting visual clinical markers
    heatmap = get_gradcam_heatmap(model, X_test[i])
    superimposed = overlay_gradcam(X_test[i], heatmap)
    actual = label_map[int(y_test[i])]
    predicted = label_map[int(predicted_labels[i])]
    prob = predictions_prob[i]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(X_test[i])
    axes[0].set_title(f'Original\nActual: {actual}')
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Overlay\nPred: {predicted} ({prob:.2f})')
    axes[2].axis('off')
    plt.suptitle(
        f'EfficientNetB0 Grad-CAM - {valid_image_names[i]}',
        fontsize=12)
    plt.tight_layout()
    # Save the generated plot to disk rather than displaying interactively
    plt.savefig(os.path.join(gradcam_dir, f'gradcam_{i}.png'), dpi=150)
    plt.close()
print(f'Grad-CAM heatmaps saved to {gradcam_dir}/')
results_df = pd.DataFrame({'Image Name': valid_image_names,
                           'actual': y_test,
                           'predicted': predicted_labels,
                           'probability': predictions_prob})
results_df.to_csv('data/efficientnet_predictions.csv', index=False)
print('Predictions saved to efficientnet_predictions.csv')
print('\nEfficientNetB0 evaluation complete!')
