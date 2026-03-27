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
# Load the dataset from the local data directory
df = pd.read_csv('data/train_dataset.csv')
image_folder = 'images_resized'
missing_images = []
for img_name in df['Image Name']:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        missing_images.append(img_name)
total_images = len(df)
missing_count = len(missing_images)
print('Total images checked:', total_images)
print('Missing images:', missing_count)
if missing_count > 0:
    print('\nMissing image files:')
    for img in missing_images[:20]:
        print(img)
else:
    print('\nAll image files exist.')
if missing_count > 0:
    pd.DataFrame(
        missing_images,
        columns=['Missing Images']).to_csv(
        'data/missing_images_report.csv',
        index=False)
    print('\nMissing image report saved to missing_images_report.csv')
