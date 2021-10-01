# Credit-Card-Fraud-Detection-with-Anomaly-Detection

# Datasets
* https://www.kaggle.com/mlg-ulb/creditcardfraud
*  https://www.kaggle.com/c/ieee-fraud-detection/overview
*  https://datahub.io/machine-learning/creditcard

# How To Use
## Step1
Place [CCFD dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) into /dataset/Kaggle_CCFD .
## Step2
```bash
python3 main.py --mode train
python3 main.py --mode test
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
