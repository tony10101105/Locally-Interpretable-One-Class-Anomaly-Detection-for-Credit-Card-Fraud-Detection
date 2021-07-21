from sklearn.svm import OneClassSVM
import csv
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import utils
import random


with open("../datasets/trainData.csv", "r") as train_file:
    train_data = csv.reader(train_file)
    train_features = []
    train_labels = []
    for row in train_data:
        train_features.append(list(map(float,row[:-1])))
        train_labels.append(float(row[-1]))

with open("../datasets/testData.csv", "r") as test_file:
    test_data = csv.reader(test_file)
    test_features = []
    test_labels = []
    for row in test_data:
        test_features.append(list(map(float,row[:-1])))
        test_labels.append(float(row[-1]))

svm = OneClassSVM(kernel='rbf', nu=0.1, gamma = "auto")
# print(test_features)
train_features = np.array(train_features)

svm.fit(train_features[random.sample(range(len(train_features)), 700)])
pred = svm.predict(test_features)


# -1 for fraud data => 1 
# 1 for normal data => 0
TP = 0
TN = 0
FP = 0
FN = 0
for idx in range(len(pred)):
    if (pred[idx] == -1 and int(test_labels[idx]) == 1):
        TP +=1
    elif (pred[idx] == -1 and int(test_labels[idx]) == 0):
        FP +=1
    elif (pred[idx] == 1 and int(test_labels[idx]) == 1):
        FN +=1
    elif (pred[idx] == 1 and int(test_labels[idx]) == 0):
        TN +=1

print('TP:', float(TP))
print('FP:', float(FP))
print('TN:', float(TN))
print('FN:', float(FN))

accuracy = utils.get_accuracy(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
recall = utils.get_recall(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
precision = utils.get_precision(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
F1_score = utils.get_F1_score(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
MCC = utils.get_MCC(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
print("accuracy: {}, recall: {}, precision: {}, F1_score: {}, MCC: {}".format(accuracy, recall, precision, F1_score, MCC))


