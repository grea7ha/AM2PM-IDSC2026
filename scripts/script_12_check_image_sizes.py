import pandas as pd
from PIL import Image
import os
df = pd.read_csv('data/Labels.csv')
if 'Unnamed: 4' in df.columns:
    df.drop(columns=['Unnamed: 4'], inplace=True)
image_folder = 'images'
df['image_path'] = df['Image Name'].apply(lambda x: os.path.join(image_folder, x))
image_sizes = []
for img_path in df['image_path'].head(20):
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            image_sizes.append(img.size)
    else:
        image_sizes.append('Missing')
print('First 20 image sizes:\n')
print(image_sizes)
unique_sizes = set([s for s in image_sizes if s != 'Missing'])
print('\nUnique image sizes detected:')
print(unique_sizes)
