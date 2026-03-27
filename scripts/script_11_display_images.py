import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
df = pd.read_csv('data/Labels.csv')
if 'Unnamed: 4' in df.columns:
    df.drop(columns=['Unnamed: 4'], inplace=True)
image_folder = 'images'
df['image_path'] = df['Image Name'].apply(lambda x: os.path.join(image_folder, x))
required_columns = ['Image Name', 'Label', 'image_path']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataset.")
quality_col = None
possible_quality_cols = ['Quality Score', 'quality_score', 'Quality', 'quality']
for col in possible_quality_cols:
    if col in df.columns:
        quality_col = col
        break
num_images = min(5, len(df))
plt.figure(figsize=(15, 8))
for i in range(num_images):
    row = df.iloc[i]
    img_path = row['image_path']
    label = row['Label']
    if quality_col is not None:
        quality_score = row[quality_col]
        title_text = f'Label: {label}\nQuality: {quality_score}'
    else:
        title_text = f'Label: {label}\nQuality: N/A'
    plt.subplot(1, num_images, i + 1)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(title_text, fontsize=10)
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=12)
        plt.title(title_text, fontsize=10)
        plt.axis('off')
plt.suptitle('First 5 Retinal Images', fontsize=16)
plt.tight_layout()
plt.show()
print('Displayed the first 5 retinal images.')
print('Observe the optic disc, retinal vessels, illumination variation, blur, and noise.')
print('These factors can affect image quality and model performance.')
