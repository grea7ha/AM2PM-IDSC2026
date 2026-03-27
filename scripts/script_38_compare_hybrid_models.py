"""
Script 38: Final Hybrid Model Comparison
=========================================
IDSC 2026 | Team AM2PM | Quality-Aware Glaucoma Triage

This is the grand finale script. It loads all three trained hybrid models
(DenseNet121, EfficientNetB0, ResNet50) and benchmarks them head-to-head
on the same held-out test set.

Key outputs:
    - A formatted console comparison table
    - hybrid_model_comparison.png     : Grouped bar chart of all metrics
    - hybrid_model_comparison_roc.png : Combined ROC curves
    - hybrid_all_metrics.json         : Machine-readable metric summary

Note on preprocessing:
    ResNet50 requires its own `preprocess_input` (mean-centred, not [0,1]).
    All other models use standard [0, 1] normalisation.
    Both preprocessed arrays are loaded separately to avoid contamination.
"""

from tensorflow.keras.applications.resnet50 import \
    preprocess_input as resnet_preprocess
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots to disk

# ── Load test dataset metadata ──────────────────────────────────────────
print('Loading test data...')
test_df = pd.read_csv('test_full_dataset.csv')
image_folder = 'images_resized'

# ── Build two versions of the test set ───────────────────────────────────────
# Version 1: [0, 1] normalisation — for DenseNet121 and EfficientNetB0
# Version 2: ResNet50's `preprocess_input` — mean-centred, channel-wise
X_test_standard = []
X_test_resnet = []
X_test_quality = []
y_test = []
valid_names = []

for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Image Name'])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img)

    X_test_standard.append(img_array / 255.0)
    X_test_resnet.append(resnet_preprocess(img_array.astype(np.float32)))
    X_test_quality.append(row['quality_normalized'])
    y_test.append(row['label_numeric'])
    valid_names.append(row['Image Name'])

X_test_standard = np.array(X_test_standard)
X_test_resnet = np.array(X_test_resnet)
X_test_quality = np.array(X_test_quality, dtype=np.float32)
y_test = np.array(y_test)

print(f'Test images loaded: {len(y_test)}')


def evaluate_hybrid_model(model_path, model_name, X_images, X_quality):
    """
    Load a trained hybrid model and compute all evaluation metrics.

    Args:
        model_path : Path to the saved .h5 model file.
        model_name : Human-readable name for reporting.
        X_images   : Preprocessed image arrays (N, 224, 224, 3).
        X_quality  : Normalised quality scores (N,).

    Returns:
        Tuple of (metrics dict, predicted probabilities array)
    """
    print(f'\nEvaluating: {model_name}...')
    model = tf.keras.models.load_model(model_path)

    predictions_prob = model.predict([X_images, X_quality]).flatten()
    predicted_labels = (predictions_prob > 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_test, predicted_labels)),
        'precision': float(precision_score(y_test, predicted_labels, zero_division=0)),
        'recall': float(recall_score(y_test, predicted_labels, zero_division=0)),
        'f1_score': float(f1_score(y_test, predicted_labels, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, predictions_prob)),
    }

    return metrics, predictions_prob


# ── Run evaluation for all three hybrid models ──────────────────────────
# (model path, display name, which preprocessed test set to use)
models_to_evaluate = [
    ('hybrid_densenet.h5', 'Hybrid DenseNet121', X_test_standard),
    ('hybrid_efficientnet.h5', 'Hybrid EfficientNetB0', X_test_standard),
    ('hybrid_resnet50.h5', 'Hybrid ResNet50', X_test_resnet),
]

all_metrics = {}
all_probs = {}

for model_path, model_name, X_images in models_to_evaluate:
    if not os.path.exists(model_path):
        print(f'WARNING: {model_path} not found — skipping {model_name}.')
        continue
    metrics, probs = evaluate_hybrid_model(
        model_path, model_name, X_images, X_test_quality)
    all_metrics[model_name] = metrics
    all_probs[model_name] = probs

# ── Print comparison table ──────────────────────────────────────────────
metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

print('\n' + '=' * 75)
print('HYBRID MODEL COMPARISON — Quality-as-Feature Glaucoma Detection')
print('=' * 75)

header = f"{'Metric':<15}" + ''.join(f'{name:<25}' for name in all_metrics)
print(header)
print('-' * len(header))

for metric in metric_names:
    row = f'{metric:<15}'
    for model_name in all_metrics:
        row += f'{all_metrics[model_name][metric]:<25.4f}'
    print(row)

print('=' * 75)

# ── Declare winner ──────────────────────────────────────────────────────
print('\nBest model per metric:')
for metric in metric_names:
    best = max(all_metrics, key=lambda m: all_metrics[m][metric])
    print(f'  {metric:<15}: {best} ({all_metrics[best][metric]:.4f})')

# Combined F1 + AUC determines the best overall model for clinical deployment
best_overall = max(
    all_metrics,
    key=lambda m: (all_metrics[m]['f1_score'], all_metrics[m]['roc_auc'])
)
print(f'\n🏆 BEST OVERALL MODEL: {best_overall}')
print(f"   F1-Score : {all_metrics[best_overall]['f1_score']:.4f}")
print(f"   ROC-AUC  : {all_metrics[best_overall]['roc_auc']:.4f}")

# ── Bar chart comparison ────────────────────────────────────────────────
colors = ['#50C878', '#4A90D9', '#E8744F']
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

for i, (model_name, metrics) in enumerate(all_metrics.items()):
    values = [metrics[m] for m in metric_names]
    bars = ax.bar(
        x + i * width,
        values,
        width,
        label=model_name,
        color=colors[i])

    # Annotate each bar with its numeric score
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=8, fontweight='bold'
        )

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Hybrid Model Comparison — Quality-as-Feature Glaucoma Detection',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
ax.legend(loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('hybrid_model_comparison.png', dpi=200)
plt.close()
print('\nBar chart saved: hybrid_model_comparison.png')

# ── Combined ROC curve ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
line_styles = ['-', '--', '-.']

for i, (model_name, probs) in enumerate(all_probs.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = all_metrics[model_name]['roc_auc']
    ax.plot(fpr, tpr,
            linestyle=line_styles[i],
            color=colors[i],
            label=f'{model_name} (AUC = {auc_val:.4f})',
            linewidth=2)

# Diagonal reference line (random classifier baseline)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison — Hybrid Quality-as-Feature Models',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hybrid_model_comparison_roc.png', dpi=200)
plt.close()
print('ROC curve saved: hybrid_model_comparison_roc.png')

# ── Persist full metrics as JSON ────────────────────────────────────────
with open('hybrid_all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print('Metrics saved: hybrid_all_metrics.json')

print('\n✅ Hybrid model comparison complete!')
