from pyod.models.ocsvm import OCSVM
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  utils

random.seed(0)

train_data = pd.read_csv("../datasets/trainData.csv", header = None).values
test_data = pd.read_csv("../datasets/testData.csv", header = None).values

train_sample_num = 700
train_data = train_data[random.sample(range(len(train_data)), train_sample_num)]


clf = OCSVM(kernel="rbf", nu=0.1, gamma = "auto")
clf.fit(train_data[:,:-1])

y_test_pred = clf.predict(test_data[:,:-1])  # outlier labels (0 or 1)
y_test_label = test_data[:,-1].astype(int).tolist()

TP = 0
TN = 0
FP = 0
FN = 0

for x in zip(y_test_pred, y_test_label):
    if x[0] == x[1]:
        if x[1] == 0:
            TN+=1
        elif x[1] == 1:
            TP+=1
    else:
        if x[1] == 0:
            FN+=1
        elif x[1] == 1:
            FP+=1

accuracy = utils.get_accuracy(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
recall = utils.get_recall(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
precision = utils.get_precision(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
F1_score = utils.get_F1_score(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
MCC = utils.get_MCC(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
print("accuracy: {}, recall: {}, precision: {}, F1_score: {}, MCC: {}".format(accuracy, recall, precision, F1_score, MCC))
