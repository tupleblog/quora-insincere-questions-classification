"""
Code to prepare K-fold cross validation
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('train.csv.zip')
kfold = StratifiedKFold(n_splits=5)
i = 0
for train_index, test_index in kfold.split(df.question_text, df.target):
    train_df, valid_df = df.iloc[train_index], df.iloc[test_index]
    train_df.to_csv('data/training_{}.csv'.format(i))
    valid_df.to_csv('data/validation_{}.csv'.format(i))
    i += 1