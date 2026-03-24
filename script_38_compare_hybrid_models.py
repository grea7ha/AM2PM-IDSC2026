import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print('Loading test data...')
test_df = pd.read_csv('test_full_dataset.csv')
image_folder = 'images_resized'
X_test_standard, X_test_quality, y_test = ([], [], [])
valid_image_names = []
for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        X_test_standard.append(img_array)
        X_test_quality.append(row['quality_normalized'])
        y_test.append(row['label_numeric'])
        valid_image_names.append(row['Image Name'])
X_test_standard = np.array(X_test_standard)
X_test_quality = np.array(X_test_quality, dtype=np.float32)
y_test = np.array(y_test)
X_test_resnet = []
for name in valid_image_names:
    img_path = os.path.join(image_folder, name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = resnet_preprocess(img_array)
    X_test_resnet.append(img_array)
X_test_resnet = np.array(X_test_resnet)
print(f'Test images: {len(y_test)}')

def evaluate_hybrid_model(model_path, model_name, X_images, X_quality):
    """Load a hybrid model and compute all metrics."""
    print(f'\nEvaluating {model_name}...')
    model = tf.keras.models.load_model(model_path)
    predictions_prob = model.predict([X_images, X_quality]).flatten()
    predicted_labels = (predictions_prob > 0.5).astype(int)
    metrics = {'accuracy': float(accuracy_score(y_test, predicted_labels)), 'precision': float(precision_score(y_test, predicted_labels, zero_division=0)), 'recall': float(recall_score(y_test, predicted_labels, zero_division=0)), 'f1_score': float(f1_score(y_test, predicted_labels, zero_division=0)), 'roc_auc': float(roc_auc_score(y_test, predictions_prob))}
    return (metrics, predictions_prob)
models_info = [('hybrid_densenet.h5', 'Hybrid DenseNet121', X_test_standard), ('hybrid_efficientnet.h5', 'Hybrid EfficientNetB0', X_test_standard), ('hybrid_resnet50.h5', 'Hybrid ResNet50', X_test_resnet)]
all_metrics = {}
all_probs = {}
for model_path, model_name, X_images in models_info:
    if os.path.exists(model_path):
        metrics, probs = evaluate_hybrid_model(model_path, model_name, X_images, X_test_quality)
        all_metrics[model_name] = metrics
        all_probs[model_name] = probs
    else:
        print(f'WARNING: {model_path} not found. Skipping {model_name}.')
print('\n' + '=' * 75)
print('HYBRID MODEL COMPARISON — Quality-as-Feature Glaucoma Detection')
print('=' * 75)
metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
header = f"{'Metric':<15}" + ''.join((f'{name:<25}' for name in all_metrics.keys()))
print(header)
print('-' * len(header))
for metric in metric_names:
    row = f'{metric:<15}'
    for model_name in all_metrics:
        value = all_metrics[model_name][metric]
        row += f'{value:<25.4f}'
    print(row)
print('=' * 75)
print('\nBest model per metric:')
for metric in metric_names:
    best_model = max(all_metrics, key=lambda m: all_metrics[m][metric])
    best_value = all_metrics[best_model][metric]
    print(f'  {metric:<15}: {best_model} ({best_value:.4f})')
best_overall = max(all_metrics, key=lambda m: (all_metrics[m]['f1_score'], all_metrics[m]['roc_auc']))
print(f'\n🏆 BEST OVERALL MODEL: {best_overall}')
print(f"   F1 Score: {all_metrics[best_overall]['f1_score']:.4f}")
print(f"   ROC-AUC:  {all_metrics[best_overall]['roc_auc']:.4f}")
x = np.arange(len(metric_names))
width = 0.25
colors = ['#50C878', '#4A90D9', '#E8744F']
fig, ax = plt.subplots(figsize=(14, 6))
for i, (model_name, metrics) in enumerate(all_metrics.items()):
    values = [metrics[m] for m in metric_names]
    bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Hybrid Model Comparison — Quality-as-Feature Glaucoma Detection', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
ax.legend(loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('hybrid_model_comparison.png', dpi=200)
plt.close()
print('\nBar chart saved to hybrid_model_comparison.png')
fig, ax = plt.subplots(figsize=(7, 6))
line_styles = ['-', '--', '-.']
for i, (model_name, probs) in enumerate(all_probs.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = all_metrics[model_name]['roc_auc']
    ax.plot(fpr, tpr, linestyle=line_styles[i], color=colors[i], label=f'{model_name} (AUC = {auc_val:.4f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison — Hybrid Quality-as-Feature Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hybrid_model_comparison_roc.png', dpi=200)
plt.close()
print('Combined ROC curve saved to hybrid_model_comparison_roc.png')
with open('hybrid_all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print('All metrics saved to hybrid_all_metrics.json')
print('\n✅ Hybrid model comparison complete!')
