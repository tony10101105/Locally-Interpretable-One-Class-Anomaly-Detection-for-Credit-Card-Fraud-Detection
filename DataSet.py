import csv
import torch
from torch.utils.data.dataset import Dataset

torch.manual_seed(0)#for reproducibility
class SplitedDataSet(Dataset):

    def __init__(self, mode = "non-fraud"):

        CREDIT_CARD_DIRECTORY = './datasets/Kaggle_CCFD/creditcard.csv'

        # data loading
        self.features = []
        self.labels = []
        csvCreditCard = open(CREDIT_CARD_DIRECTORY)
        CreditCardData = csv.reader(csvCreditCard)
        
        if mode == 'non-fraud':
            skipped_class = '1'
        elif mode == 'fraud':
            skipped_class = '0'

        for row in CreditCardData:
            if row[-1] == skipped_class or row[-1] == 'Class':
                continue
            self.features.append(row[:-1])
            self.labels.append(row[-1])
            
        # convert elements from string to float
        for i in range(len(self.features)):
            self.features[i] = list(map(float, self.features[i]))
        self.labels = list(map(float, self.labels))
        
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        assert len(self.features) == len(self.labels), print('data length error')
        return len(self.features)

class DataSet(Dataset):
    def __init__(self, datasets = []):
        self.features = []
        self.labels = []

        for dataset in datasets:
            self.features += [dataset[i][0][1:-1] for i in range(len(dataset))]
            self.labels += [dataset[i][1] for i in range(len(dataset))]
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        assert len(self.features) == len(self.labels), print('data length error')
        return len(self.features)