"""
Script 18: Patient-Level Train/Test Split
==========================================
IDSC 2026 | Team AM2PM | Quality-Aware Glaucoma Triage

This script performs a strict patient-level train/test split to prevent
data leakage. Since each patient may have images of both eyes, a naive
random image-level split would allow the model to memorize patient-specific
anatomy rather than learning true disease patterns.

By splitting on unique Patient IDs, we guarantee zero overlap between
training and testing populations.

Output:
    - train_dataset.csv : Training set metadata
    - test_dataset.csv  : Testing set metadata
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Load and clean the dataset ──────────────────────────────────────────
df = pd.read_csv('data/Labels.csv')

# Remove any spurious empty columns from the CSV export
if 'Unnamed: 4' in df.columns:
    df = df.drop(columns=['Unnamed: 4'])

# ── Quality filter ──────────────────────────────────────────────────────
# Keep only images with a Quality Score >= 5 for the baseline pipeline.
# The hybrid models (script_31+) will use all 747 images without this filter.
df = df[df['Quality Score'] >= 5]
print(f'Images passing quality filter (score >= 5): {len(df)}')

# ── Encode labels as binary integers ─────────────────────────────────────────
df['label_numeric'] = df['Label'].map({'GON+': 1, 'GON-': 0})

# ── Patient-level split ─────────────────────────────────────────────────
# Split on unique patient IDs, not on individual images.
# random_state=42 ensures full reproducibility.
patients = df['Patient'].unique()
train_patients, test_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

# Each image is assigned to the split that its patient belongs to
train_df = df[df['Patient'].isin(train_patients)]
test_df = df[df['Patient'].isin(test_patients)]

# ── Report split statistics ─────────────────────────────────────────────
print(
    f'\nTraining set: {len(train_df)} images across {train_df["Patient"].nunique()} unique patients')
print(
    f'Testing set : {len(test_df)} images across {test_df["Patient"].nunique()} unique patients')

# ── Save the splits ─────────────────────────────────────────────────────
train_df.to_csv('data/train_dataset.csv', index=False)
test_df.to_csv('data/test_dataset.csv', index=False)
print('\nSaved: train_dataset.csv, test_dataset.csv')
