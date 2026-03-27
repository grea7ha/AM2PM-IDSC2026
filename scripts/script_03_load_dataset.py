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
print('Dataset successfully loaded.\n')
print('First 5 rows of dataset:\n')
print(df.head())
print('\nDataset shape:')
print(df.shape)
print('\nColumn names:')
print(df.columns)
