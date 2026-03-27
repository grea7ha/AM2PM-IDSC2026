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
train_df = pd.read_csv('data/train_dataset.csv')
# Load the dataset from the local data directory
test_df = pd.read_csv('data/test_dataset.csv')
print('Training label distribution:')
print(train_df['Label'].value_counts())
print('\nTesting label distribution:')
print(test_df['Label'].value_counts())
