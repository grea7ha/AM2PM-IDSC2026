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
import os
from PIL import Image
# Load the dataset from the local data directory
df = pd.read_csv('data/Labels.csv')
if 'Unnamed: 4' in df.columns:
    # Clean up any spurious empty columns from Excel/CSV export
    df = df.drop(columns=['Unnamed: 4'])
image_folder = 'images'
resized_folder = 'images_resized'
if not os.path.exists(resized_folder):
    os.makedirs(resized_folder)
for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row['Image Name'])
    # Open the image file and convert to RGB format
    img = Image.open(image_path)
    # Resize standard dimensions suitable for ImageNet pre-trained backbones
    img = img.resize((224, 224))
    save_path = os.path.join(resized_folder, row['Image Name'])
    img.save(save_path)
print('All images resized successfully.')
