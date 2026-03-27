import pandas as pd
df = pd.read_csv('data/Labels.csv')
print('Label distribution:\n')
print(df['Label'].value_counts())
