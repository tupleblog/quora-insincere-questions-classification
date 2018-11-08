"""
Code to prepare K-fold cross validation
"""
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def prepare_kfold_dataset(path='train.csv.zip', n_splits=5):
    df = pd.read_csv(path)
    kfold = StratifiedKFold(n_splits=n_splits)
    i = 0
    for train_index, test_index in kfold.split(df.question_text, df.target):
        train_df, valid_df = df.iloc[train_index], df.iloc[test_index]
        train_df.to_csv('data/training_{}.csv'.format(i))
        valid_df.to_csv('data/validation_{}.csv'.format(i))
        i += 1
    print('done!')


def prepare_experiments(n_splits=5):
    params = json.load(open('experiments/quora.json'))
    for i in range(n_splits):
        params['train_data_path'] = 'data/training_{}.csv'.format(i)
        params['validation_data_path'] = 'data/validation_{}.csv'.format(i)
        with open('experiments/quora_{}.json'.format(i), 'w') as f:
            json.dump(params, f, indent=4)


if __name__ == '__main__':
    prepare_kfold_dataset(path='train.csv.zip', n_splits=5)
    prepare_experiments(n_splits=5)