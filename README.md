# Locally Interpretable One-Class Anomaly Detection for Credit Card Fraud Detection
Official code of the paper [Locally Interpretable One-Class Anomaly Detection for Credit Card Fraud Detection](https://arxiv.org/abs/2108.02501)

## Datasets
* https://www.kaggle.com/mlg-ulb/creditcardfraud

## :rocket: Usage
1. Place [CCFD dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) into */datasets/Kaggle_CCFD*.
2. Run:
```bash
python3 main.py --mode train
python3 main.py --mode test
python3 main.py --mode explainer
```

## The Pyod Experiments
1. Use function ```writeToCsv()``` in ```split.py``` to split dataset into training set and testing set.
2. Run the ```.py``` in */pyod*, such as:
```
cd pyod
python3 ocnn.py
```
