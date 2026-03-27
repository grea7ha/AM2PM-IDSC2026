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
import pandas as pd
# Load the dataset from the local data directory
df = pd.read_csv('data/Labels.csv')
if 'Unnamed: 4' in df.columns:
    # Clean up any spurious empty columns from Excel/CSV export
    df = df.drop(columns=['Unnamed: 4'])
df = df[df['Quality Score'] >= 5]
print('Filtered dataset size:', len(df))
df['label_numeric'] = df['Label'].map({'GON+': 1, 'GON-': 0})
df.to_csv('data/glaucoma_clean_dataset.csv', index=False)
print('Clean dataset saved.')
