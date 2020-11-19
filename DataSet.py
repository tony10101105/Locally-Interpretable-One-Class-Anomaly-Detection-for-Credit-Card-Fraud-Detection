import csv
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np

torch.manual_seed(0)#for reproducibility

def z_score_normalization(x):
    x = np.asarray(x, dtype = np.float)
    for i in range(len(x[0])):  
        mean = np.mean(x[:,i])
        std = np.std(x[:,i])
        x[:,i] = (x[:,i] - mean) / std
    return x

def mix_max_normalization(x):
    x = np.asarray(x, dtype = np.float)
    for i in range(len(x[0])):  
        max = x[:,i].max()
        min = x[:,i].min()
        x[:,i] = (x[:,i] - min) / (max - min)
    return x


class DataSet(Dataset):

    def __init__(self, mode = 'non-fraud', normalization_type = 'mix_max'):

        if mode == 'non-fraud':
            CREDIT_CARD_DIRECTORY = './datasets/creditcard_0.csv'
        elif mode == 'fraud':
            CREDIT_CARD_DIRECTORY = './datasets/creditcard_1.csv'

        ##preprocessing of numerical data
        self.features = []
        self.labels = []
        csvCreditCard = open(CREDIT_CARD_DIRECTORY)
        CreditCardData = csv.reader(csvCreditCard)
        
        for row in CreditCardData:
            self.features.append(row[:-1])
            self.labels.append(row[-1])
            
        #####convert elements from string to float
        for i in range(len(self.features)):
            self.features[i] = list(map(float, self.features[i]))
        self.labels = list(map(float, self.labels))
        #####
        
        if normalization_type == 'mix_max':
            self.features = mix_max_normalization(self.features)
        elif normalization_type == 'z_score':
            self.features = z_score_normalization(self.features)
        else:
            raise Exception('this type of normalization not implemented yet')

        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        assert len(self.features) == len(self.labels), print('data length error')
        return len(self.features)

