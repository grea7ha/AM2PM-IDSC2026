import pandas as pd
import os
df = pd.read_csv('train_dataset.csv')
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
    pd.DataFrame(missing_images, columns=['Missing Images']).to_csv('missing_images_report.csv', index=False)
    print('\nMissing image report saved to missing_images_report.csv')
