import DataSet
import torch
from torch.utils.data import random_split
import csv
import random

torch.manual_seed(4)#for reproducibility
random.seed(0)

def getDatasets():
    non_fraud_Data = DataSet.SplitedDataSet(mode = 'non-fraud')
    fraud_Data = DataSet.SplitedDataSet(mode = 'fraud')

    data_point_num = len(non_fraud_Data)
    test_data_point_num = 490
    train_data_point_num = data_point_num - test_data_point_num
    trainData, nonFraudTestData = random_split(non_fraud_Data, [train_data_point_num, test_data_point_num])

    trainData = DataSet.DataSet([trainData])
    fraud_Data, _ = random_split(fraud_Data, [490, 2])
    testData = DataSet.DataSet([nonFraudTestData, fraud_Data]) #following the setting of 13.pdf

    return trainData, testData

def writeToCsv():
    trainData, testData = getDatasets()
    with open("./datasets/trainData.csv", "w") as f:
        writer = csv.writer(f)

        for i in range(len(trainData)):
            writer.writerow(trainData[i][0]+[trainData[i][1]])

    with open("./datasets/testData.csv", "w") as f:
        writer = csv.writer(f)

        for i in range(len(testData)):
            writer.writerow(testData[i][0]+[testData[i][1]])
