# Credit-Card-Fraud-Detection-with-Anomaly-Detection
> This is official code implementation of TAAI 2021 Best Paper Award paper: Locally Interpretable One Class Anomaly Detection for Credit Card Fraud Detection.
# Datasets
* https://www.kaggle.com/mlg-ulb/creditcardfraud

# How To Use
## Step1
Place [CCFD dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) into /datasets/Kaggle_CCFD .
## Step2
```bash
python3 main.py --mode train
python3 main.py --mode test
python3 main.py --mode explainer
```

# Run Pyod Experiments
## Step1
Use function ```writeToCsv()``` in ```split.py``` to split dataset into training set and testing set.
## Step2
Run the ```.py``` in /pyod directly, eg.
```
cd pyod
python3 ocnn.py
```

