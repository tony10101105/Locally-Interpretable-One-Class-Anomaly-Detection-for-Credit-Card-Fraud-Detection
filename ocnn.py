import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import utils

random.seed(0)

train_data = pd.read_csv("./datasets/trainData.csv", header = None).values
test_data = pd.read_csv("./datasets/testData.csv", header = None).values

train_sample_num = 700
train_data = train_data[random.sample(range(len(train_data)), train_sample_num)]

# data = np.array(train_data.tolist()+test_data.tolist())
# data = np.array(test_data.tolist())

nbrs = NearestNeighbors(n_neighbors = 20)
nbrs.fit(train_data[:,:-1])

distances, indexes = nbrs.kneighbors(test_data[:,:-1])

# print(distances)

plt.figure()
plt.scatter( range(len(distances)), distances.mean(axis =1), s=9)
plt.savefig('./fig/mean')

TP = 0
TN = 0
FP = 0
FN = 0

highest_MCC = 0
params = []

for threshold in range(0, 10*int(max(distances.mean(axis=1)))):
    outlier_index = np.where(distances.mean(axis = 1) > threshold/10)[0]
    TP = len(np.where(outlier_index >=  len(test_data)/2)[0])
    FP = len(outlier_index) - TP
    TN = len(test_data)/2 - FP
    FN = len(test_data)/2 - TP
    # print(f"======= Threshold = {threshold/2} =======")
    # print('TP:', float(TP))
    # print('FP:', float(FP)) 
    # print('TN:', float(TN))
    # print('FN:', float(FN))
    MCC = utils.get_MCC(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
    if MCC > highest_MCC:
        accuracy = utils.get_accuracy(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        recall = utils.get_recall(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        precision = utils.get_precision(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        F1_score = utils.get_F1_score(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        params = [threshold/10, accuracy, recall, precision, F1_score, MCC]
        highest_MCC = MCC
    else:
        continue
    
    
    # print("accuracy: {}, recall: {}, precision: {}, F1_score: {}, MCC: {}".format(accuracy, recall, precision, F1_score, MCC))
    # print()

print("threshold:{},accuracy: {}, recall: {}, precision: {}, F1_score: {}, MCC: {}".format(params[0],params[1], params[2], params[3], params[4], params[5]))