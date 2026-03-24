import pandas as pd
df = pd.read_csv('C:\\Users\\Thanush\\Downloads\\glaucoma_bootcamp\\Labels.csv')
print('Dataset successfully loaded.\n')
print('First 5 rows of dataset:\n')
print(df.head())
print('\nDataset shape:')
print(df.shape)
print('\nColumn names:')
print(df.columns)
