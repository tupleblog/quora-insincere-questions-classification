# Quora Insincere Questions Classification

Code for Kaggle competition [here](https://www.kaggle.com/c/quora-insincere-questions-classification)


## Prepare dataset

Put `train.csv.zip` in this folder and run `python kfold_validation_data.py` to generate K-folds dataset and experiments file


## Run all models

This will run `allennlp` on all generated K-folds and put into `output_{}` folders

```bash
bash run_experiments.sh
```