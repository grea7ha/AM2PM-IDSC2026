import pandas as pd
train_df = pd.read_csv('data/train_dataset.csv')
test_df = pd.read_csv('data/test_dataset.csv')
print('Training label distribution:')
print(train_df['Label'].value_counts())
print('\nTesting label distribution:')
print(test_df['Label'].value_counts())
