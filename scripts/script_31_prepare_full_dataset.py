"""
Script 31: Prepare Full Dataset (Quality-Aware Pipeline)
=========================================================
IDSC 2026 | Team AM2PM | Quality-Aware Glaucoma Triage

Unlike the baseline pipeline (script_18), which filtered out low-quality
images, this script deliberately retains ALL 747 images — including those
with degraded quality scores.

The key innovation: instead of discarding "bad" images, we normalise the
Quality Score to [0, 1] and pass it as a direct input feature to our
hybrid neural network. This teaches the model to mathematically account
for image quality when making its prediction.

Output:
    - train_full_dataset.csv : Full training set (no quality filter)
    - test_full_dataset.csv  : Full testing set (no quality filter)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Load and clean the dataset ────────────────────────────────────────────────
df = pd.read_csv('data/Labels.csv')

if 'Unnamed: 4' in df.columns:
    df = df.drop(columns=['Unnamed: 4'])

print(f'Full dataset size (no quality filter): {len(df)} images')

# ── Feature engineering ───────────────────────────────────────────────────────
# Encode the categorical label as a binary integer target
df['label_numeric'] = df['Label'].map({'GON+': 1, 'GON-': 0})

# Normalise the Quality Score from its raw range to [0, 1]
# This ensures numerical parity with the image pixel values (also [0, 1])
df['quality_normalized'] = df['Quality Score'] / 10.0

# ── Patient-level split ───────────────────────────────────────────────────────
# Identical strategy to script_18: split patients, not images.
# Using the same random_state=42 for consistency across experiments.
patients = df['Patient'].unique()
train_patients, test_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

train_df = df[df['Patient'].isin(train_patients)]
test_df  = df[df['Patient'].isin(test_patients)]

# ── Report split statistics ───────────────────────────────────────────────────
print(f'\nTraining set : {len(train_df)} images across {train_df["Patient"].nunique()} unique patients')
print(f'Testing set  : {len(test_df)} images across {test_df["Patient"].nunique()} unique patients')

print('\nQuality Score distribution in training set:')
print(train_df['Quality Score'].describe().round(3))

# ── Save the splits ───────────────────────────────────────────────────────────
train_df.to_csv('data/train_full_dataset.csv', index=False)
test_df.to_csv('data/test_full_dataset.csv', index=False)
print('\nSaved: train_full_dataset.csv, test_full_dataset.csv')
