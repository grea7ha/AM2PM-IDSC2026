import pandas as pd
df = pd.read_csv('data/Labels.csv')
print('Quality score statistics:\n')
print(df['Quality Score'].describe())
