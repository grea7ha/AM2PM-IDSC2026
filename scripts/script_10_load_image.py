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
from PIL import Image
import os
# Load the dataset from the local data directory
df = pd.read_csv('data/Labels.csv')
if 'Unnamed: 4' in df.columns:
    # Clean up any spurious empty columns from Excel/CSV export
    df = df.drop(columns=['Unnamed: 4'])
    df['image_path'] = df['Image Name'].apply(
        lambda x: os.path.join('images', x))
image_path = df.iloc[0]['image_path']
# Open the image file and convert to RGB format
img = Image.open(image_path)
print('Image path:', image_path)
print('Image size:', img.size)
print('Image mode:', img.mode)
