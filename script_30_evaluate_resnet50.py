import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
print('Loading ResNet50 model...')
model = tf.keras.models.load_model('glaucoma_model_resnet50.h5')
test_df = pd.read_csv('test_dataset.csv')
image_folder = 'images_resized'
X_test, y_test = ([], [])
valid_image_names = []
for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        X_test.append(img_array)
        y_test.append(row['label_numeric'])
        valid_image_names.append(row['Image Name'])
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f'Test images loaded: {len(X_test)}')
predictions_prob = model.predict(X_test).flatten()
predicted_labels = (predictions_prob > 0.5).astype(int)
metrics = {'accuracy': float(accuracy_score(y_test, predicted_labels)), 'precision': float(precision_score(y_test, predicted_labels, zero_division=0)), 'recall': float(recall_score(y_test, predicted_labels, zero_division=0)), 'f1_score': float(f1_score(y_test, predicted_labels, zero_division=0)), 'roc_auc': float(roc_auc_score(y_test, predictions_prob))}
print('\n===== ResNet50 Evaluation Metrics =====')
for metric_name, value in metrics.items():
    print(f'  {metric_name:>12s}: {value:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, predicted_labels, target_names=['Normal', 'Glaucoma']))
with open('resnet50_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics saved to resnet50_metrics.json')
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Glaucoma'], yticklabels=['Normal', 'Glaucoma'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('ResNet50 - Confusion Matrix')
plt.tight_layout()
plt.savefig('resnet50_cm.png', dpi=150)
plt.close()
print('Confusion matrix saved to resnet50_cm.png')
fpr, tpr, _ = roc_curve(y_test, predictions_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ResNet50 (AUC = {metrics['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ResNet50 - ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('resnet50_roc.png', dpi=150)
plt.close()
print('ROC curve saved to resnet50_roc.png')
print('\nGenerating Grad-CAM heatmaps...')

def get_gradcam_heatmap(img_array, sub_base_model, head_layers):
    """Generate Grad-CAM heatmap for a given image using manual path bridging."""
    img_tensor = tf.cast(tf.expand_dims(img_array, axis=0), tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, base_output = sub_base_model(img_tensor)
        x = base_output
        for layer in head_layers:
            x = layer(x)
        predictions = x
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_vals = conv_outputs[0]
    heatmap = conv_outputs_vals @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-08)
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img = np.uint8(img_array)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(Image.fromarray(heatmap_resized).resize((img.shape[1], img.shape[0])))
    colormap = plt.cm.jet
    heatmap_colored = colormap(heatmap_resized / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    superimposed = np.uint8(heatmap_colored * alpha + img * (1 - alpha))
    return superimposed
gradcam_dir = 'gradcam_resnet50'
os.makedirs(gradcam_dir, exist_ok=True)
print('Preparing Grad-CAM components...')
try:
    base_model = model.get_layer('resnet50')
    target_layer = base_model.get_layer('conv5_block3_out')
    sub_base_model = tf.keras.Model(inputs=base_model.inputs, outputs=[target_layer.output, base_model.output])
    head_layers = []
    base_idx = -1
    for i, layer in enumerate(model.layers):
        if layer == base_model:
            base_idx = i
            break
    if base_idx != -1:
        head_layers = model.layers[base_idx + 1:]
    print(f'Path bridged: {base_model.name} -> {len(head_layers)} head layers.')
except Exception as e:
    print(f'Error preparing Grad-CAM: {e}')
    sub_base_model = None
    head_layers = []
label_map = {1: 'Glaucoma', 0: 'Normal'}
num_samples = min(5, len(X_test))
X_vis = []
for i in range(num_samples):
    img_name = valid_image_names[i]
    img_path = os.path.join(image_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    X_vis.append(np.array(img))
for i in range(num_samples):
    if sub_base_model and head_layers:
        heatmap = get_gradcam_heatmap(X_test[i], sub_base_model, head_layers)
    else:
        heatmap = np.zeros((7, 7), dtype=np.float32)
    superimposed = overlay_gradcam(X_vis[i], heatmap)
    actual = label_map[int(y_test[i])]
    predicted = label_map[int(predicted_labels[i])]
    prob = predictions_prob[i]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(X_vis[i])
    axes[0].set_title(f'Original\nActual: {actual}')
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Overlay\nPred: {predicted} ({prob:.2f})')
    axes[2].axis('off')
    plt.suptitle(f'ResNet50 Grad-CAM - {valid_image_names[i]}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(gradcam_dir, f'gradcam_{i}.png'), dpi=150)
    plt.close()
print(f'Grad-CAM heatmaps saved to {gradcam_dir}/')
results_df = pd.DataFrame({'Image Name': valid_image_names, 'actual': y_test, 'predicted': predicted_labels, 'probability': predictions_prob})
results_df.to_csv('resnet50_predictions.csv', index=False)
print('Predictions saved to resnet50_predictions.csv')
print('\nResNet50 evaluation complete!')
