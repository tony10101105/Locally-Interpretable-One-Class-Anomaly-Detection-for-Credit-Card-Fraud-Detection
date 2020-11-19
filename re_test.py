import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
import models
import DataSet
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--lr", type = float, default = 2e-5)
parser.add_argument("--n_epochs", type = int, default = 50)
parser.add_argument("--normalization", type = str, default = 'mix_max')
parser.add_argument("--reconstructionLoss", type = str, default = 'L1')
parser.add_argument("--threshold", type = float, default = 0.7)
parser.add_argument("--mode", type = str, default = 'test')
parser.add_argument("--GPU", type = bool, default = False)
parser.add_argument("--test_ratio", type = float, default = 0.2)
parser.add_argument("--seed", type = int, default = 0)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)#for reproducibility

#load datasets
print('loading the {} dataset...'.format(args.mode+'ing'))
non_fraud_Data = DataSet.DataSet(mode = 'non-fraud', normalization_type = args.normalization)
fraud_Data = DataSet.DataSet(mode = 'fraud', normalization_type = args.normalization)
testData = fraud_Data
'''data_point_num = len(non_fraud_Data)
test_data_point_num = int(data_point_num * args.test_ratio)
train_data_point_num = data_point_num - test_data_point_num
trainData, testData = random_split(non_fraud_Data, [train_data_point_num, test_data_point_num])
testData = ConcatDataset([testData, fraud_Data])'''

#trainDataLoader = DataLoader(dataset = trainData, batch_size = args.batch_size, shuffle = True)
testDataLoader = DataLoader(dataset = testData, batch_size = args.batch_size, shuffle = True)
print('datasets loading finished!')


#load models
if os.path.exists('./checkpoints/g_checkpoint.pth') and os.path.exists('./checkpoints/d_checkpoint.pth'):
    print('loading existing (pretrained) models...')
    g_path = './checkpoints/g_checkpoint.pth'
    d_path = './checkpoints/d_checkpoint.pth'
    
    generator, discriminator, g_optimizer, d_optimizer, current_epoch = utils.load_checkpoint(g_path, d_path)
    
else:
    generator = models.autoencoder()
    discriminator = models.FCNN()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr = args.lr, weight_decay = 1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = args.lr, weight_decay = 1e-3)
    current_epoch = 0


generator.eval()
discriminator.eval()

if args.GPU == True and torch.cuda.is_available():
    print('using GPU...')
    generator = generator.cuda()
    discriminator = discriminator.cuda()

if args.reconstructionLoss == 'MSE':
    reconstructionLoss = nn.MSELoss()
elif args.reconstructionLoss == 'L1':
    reconstructionLoss = nn.L1Loss()
elif args.reconstructionLoss == 'SmoothL1':
    reconstructionLoss = nn.SmoothL1Loss()
else:
    raise Exception('loss function setting error')

#variables for recording losses and accuracy/MCC
g_loss_Re =0
g_loss_BCE =0
d_loss_sum = 0
TP, FP, FN, TN = 0, 0, 0, 0#4 elements of confusion metrix for calculating MCC

for i, (features, labels) in enumerate(testDataLoader):
            
    labels = labels.unsqueeze(1)

    if args.GPU == True and torch.cuda.is_available():
        features = features.cuda()
        labels = labels.cuda()

    print('labels:', labels.data)
    print('original:', features.data)
    reconstructed_features = generator(features)
    print('reconstructed:', reconstructed_features.data)
    print('error:', reconstructionLoss(reconstructed_features, features))
    myerror = torch.sum(abs(reconstructed_features - features))
    print('myerror:', myerror)
    
        
        





        
