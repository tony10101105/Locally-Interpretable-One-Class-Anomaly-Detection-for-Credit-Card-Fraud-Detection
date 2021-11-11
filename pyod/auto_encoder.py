import torch
import torch.nn as nn
from pyod.models.auto_encoder_torch import AutoEncoder
import random
import pandas as pd
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 5)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--n_epochs", type = int, default = 1)
parser.add_argument("--reconstructionLoss", type = str, default = 'MSE')
args = parser.parse_args()
print(args)

if args.reconstructionLoss == 'MSE':
    reconstructionLoss = nn.MSELoss()
elif args.reconstructionLoss == 'L1':
    reconstructionLoss = nn.L1Loss()
elif args.reconstructionLoss == 'SmoothL1':
    reconstructionLoss = nn.SmoothL1Loss()
elif args.reconstructionLoss == 'BCE':
    reconstructionLoss = nn.BCEWithLogitsLoss()
else:
    raise Exception('loss function setting error')

torch.manual_seed(4)#for reproducibility
random.seed(0)

train_data = pd.read_csv("../datasets/trainData.csv", header = None).values
test_data = pd.read_csv("../datasets/testData.csv", header = None).values

train_sample_num = 700
train_data = train_data[random.sample(range(len(train_data)), train_sample_num)]

clf = AutoEncoder(hidden_neurons=[15, 8, 15], hidden_activation="relu", batch_norm=True,learning_rate=args.lr, epochs=args.n_epochs, batch_size=args.batch_size, dropout_rate=0, weight_decay=1e-3, contamination=(len(test_data)/2)/(len(test_data)+train_sample_num))

clf.fit(train_data[:,:-1])

y_test_pred = clf.predict(test_data[:,:-1])  # outlier labels (0 or 1)
y_test_label = test_data[:,-1].astype(int).tolist()

# print(y_test_pred)

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