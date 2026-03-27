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
import matplotlib.pyplot as plt
# Load the dataset from the local data directory
df = pd.read_csv('data/Labels.csv')
label_counts = df['Label'].value_counts()
label_counts.plot(kind='bar')
plt.title('Glaucoma Label Distribution')
plt.xlabel('Label')
plt.ylabel('Number of Images')
plt.savefig('results/label_distribution.png')
# plt.show()
