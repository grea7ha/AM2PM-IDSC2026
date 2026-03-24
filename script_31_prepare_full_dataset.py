import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('Labels.csv')
if 'Unnamed: 4' in df.columns:
    df = df.drop(columns=['Unnamed: 4'])
print('Full dataset size:', len(df))
df['label_numeric'] = df['Label'].map({'GON+': 1, 'GON-': 0})
df['quality_normalized'] = df['Quality Score'] / 10.0
patients = df['Patient'].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
train_df = df[df['Patient'].isin(train_patients)]
test_df = df[df['Patient'].isin(test_patients)]
print(f'Training images: {len(train_df)}')
print(f'Testing images: {len(test_df)}')
print(f"Training patients: {train_df['Patient'].nunique()}")
print(f"Testing patients: {test_df['Patient'].nunique()}")
print(f'\nQuality Score stats (train):')
print(train_df['Quality Score'].describe())
train_df.to_csv('train_full_dataset.csv', index=False)
test_df.to_csv('test_full_dataset.csv', index=False)
print('\nDatasets saved: train_full_dataset.csv, test_full_dataset.csv')
