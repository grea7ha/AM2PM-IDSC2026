import pandas as pd
df = pd.read_csv('C:\\Users\\Thanush\\Downloads\\glaucoma_bootcamp\\Labels.csv')
print('Label distribution:\n')
print(df['Label'].value_counts())
