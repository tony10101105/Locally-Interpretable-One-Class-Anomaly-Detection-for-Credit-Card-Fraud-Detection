import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn import metrics
import argparse
import models
import DataSet
import utils
import random


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 512)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--n_epochs", type = int, default = 1)
parser.add_argument("--normalization", type = str, default = 'min_max')
parser.add_argument("--reconstructionLoss", type = str, default = 'MSE')
parser.add_argument("--mode", type = str, default = 'train')
parser.add_argument("--GPU", type = bool, default = False)
parser.add_argument("--resume", type = bool, default = False)
args = parser.parse_args()
print(args)

torch.manual_seed(4)#for reproducibility
random.seed(0)#for reproducibility

#load datasets
print('loading the dataset...')
non_fraud_Data = DataSet.DataSet(mode = 'non-fraud', normalization_type = args.normalization)
fraud_Data = DataSet.DataSet(mode = 'fraud', normalization_type = args.normalization)

data_point_num = len(non_fraud_Data)
test_data_point_num = 490
train_data_point_num = data_point_num - test_data_point_num
trainData, nonFraudTestData = random_split(non_fraud_Data, [train_data_point_num, test_data_point_num])
fraud_Data, _ = random_split(fraud_Data, [490, 2])
testData = ConcatDataset([nonFraudTestData, fraud_Data])#refer to the setting of 13.pdf

trainDataLoader = DataLoader(dataset = trainData, batch_size = args.batch_size, shuffle = True, drop_last=True)
testDataLoader = DataLoader(dataset = testData, batch_size = args.batch_size, shuffle = True)
print('datasets loading finished!')


#load models
if args.resume or args.mode == 'test':
    try:
        print('loading existing (pretrained) models...')
        g_path = './checkpoints/g_checkpoint.pth'
        d_path = './checkpoints/d_checkpoint.pth'
    
        generator, discriminator, g_optimizer, d_optimizer, current_epoch = utils.load_checkpoint(g_path, d_path)
    except:
        raise Exception('failed to load models!')
    
else:
    print('building new models...')
    generator = models.autoencoder()
    discriminator = models.FCNN()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr = args.lr, weight_decay = 1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = args.lr, weight_decay = 1e-3)
    current_epoch = 0

#some checks
if args.GPU == True and torch.cuda.is_available():
    print('using GPU...')
    generator = generator.cuda()
    discriminator = discriminator.cuda()

#come checks
if current_epoch >= args.n_epochs and args.mode == 'train':
    raise Exception('epoch number error!')
if args.mode == 'test' and (os.path.exists('./checkpoints/g_checkpoint.pth') and os.path.exists('./checkpoints/d_checkpoint.pth')) == False:
    raise Exception('no pretrained models for testing stage')

#set the loss function
if args.reconstructionLoss == 'MSE':
    reconstructionLoss = nn.MSELoss()
elif args.reconstructionLoss == 'L1':
    reconstructionLoss = nn.L1Loss()
elif args.reconstructionLoss == 'SmoothL1':
    reconstructionLoss = nn.SmoothL1Loss()
else:
    raise Exception('loss function setting error')

#set adversarial loss
BCELoss = nn.BCELoss()

#start training / testing
if args.mode == 'train':
    generator.train()
    discriminator.train()
    print('start running on training mode...')
    for epoch in range(current_epoch, args.n_epochs):
        print('epoch:', epoch + 1)
        #for recording losses and accuracy/MCC
        g_loss_Re =0
        g_loss_BCE =0
        d_loss_sum = 0
        for i, (features, labels) in enumerate(trainDataLoader):
            #noise = torch.randn_like(features)
            #noisy_features = features + noise*0.2

            labels = labels.unsqueeze(1)
            if torch.sum(labels) != 0:
                raise Exception('stop')
                
            real_label = torch.ones(labels.size())
            fake_label = torch.zeros(labels.size())

            if args.GPU == True and torch.cuda.is_available():
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                features = features.cuda()
                #noisy_features = noisy_features.cuda()
                labels = labels.cuda()

            ##train Generator
            reconstruction = generator(features)
            Re_Loss = reconstructionLoss(reconstruction, features)
            fake_pred = discriminator(reconstruction)
            
            BCE_Loss = BCELoss(fake_pred, real_label)
            g_loss = Re_Loss + BCE_Loss
            
            g_optimizer.zero_grad()
            g_loss.backward()
        
            g_optimizer.step()

            g_loss_Re += torch.sum(Re_Loss)
            g_loss_BCE += torch.sum(BCE_Loss)
        
            ##train discriminator

            real_pred = discriminator(features)
            real_loss = BCELoss(real_pred, real_label)

            fake_pred = discriminator(reconstruction.detach())
            fake_loss = BCELoss(fake_pred, fake_label)

            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_loss_sum += torch.sum(d_loss)


            if (i + 1) % 500 == 0:
                print("iteration: {} / {}, Epoch: {} / {}, g_loss_Re: {:.5f}, g_loss_BCE: {:.4f}, d_loss: {:.4f}".format(
                    str((i+1)*args.batch_size), str(train_data_point_num), epoch+1, args.n_epochs, g_loss_Re.data / (500*args.batch_size), g_loss_BCE.data / (500*args.batch_size), d_loss_sum.data / (500*args.batch_size)))
                g_loss_Re = 0
                g_loss_BCE = 0
                d_loss_sum = 0
                #print('real_pred:', real_pred)
                #print('fake_pred:', fake_pred)

        torch.save({'epoch': epoch+1, 'model_state_dict': generator.state_dict(), 'optimizer_state_dict': g_optimizer.state_dict()}, './checkpoints/g_checkpoint.pth')
        torch.save({'epoch': epoch+1, 'model_state_dict': discriminator.state_dict(), 'optimizer_state_dict': d_optimizer.state_dict()}, './checkpoints/d_checkpoint.pth')


elif args.mode == 'test':
    generator.eval()
    discriminator.eval()
    print('start running on test mode...')
    for i in range(1, 20):
        TP, FP, TN, FN = 0, 0, 0, 0
        all_pred = []
        all_labels = []
        threshold = i / 20
        print('threshold:', threshold)
        for i, (features, labels) in enumerate(testDataLoader):
            #noise = torch.randn_like(features)
            #noisy_features = features + noise*0.2

            if args.GPU == True and torch.cuda.is_available():
                features = features.cuda()
                #noisy_features = noisy_features.cuda()
                labels = labels.cuda()

            ##test Discriminator
            reconstructed_features = generator(features)
            p_fraud = discriminator(reconstructed_features)
            #print(torch.sum(features - reconstructed_features, 1))
            #print(labels)
            #p_fraud = 1 - p_fraud
            p_fraud = p_fraud.squeeze()
            print(p_fraud)
            print(labels)

            all_pred.extend(p_fraud.tolist())
            all_labels.extend(labels.tolist())
            
            final_pred = torch.zeros(p_fraud.size())
            final_pred[p_fraud >= threshold] = 1
        
            TP += torch.sum((final_pred == 1) & (labels == 1))
            FP += torch.sum((final_pred == 1) & (labels == 0))
            TN += torch.sum((final_pred == 0) & (labels == 0))
            FN += torch.sum((final_pred == 0) & (labels == 1))
        
        print('TP:', float(TP))
        print('FP:', float(FP))
        print('TN:', float(TN))
        print('FN:', float(FN))

        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_pred)
        #print('thresholds:', thresholds)
        #print('fpr:', fpr)
        #print('tpr:', tpr)
        roc_auc = metrics.auc(fpr, tpr)
        #print('roc_auc:', roc_auc)

        utils.plot_and_save_fig(fpr, tpr, roc_auc)

        accuracy = utils.get_accuracy(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        recall = utils.get_recall(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        precision = utils.get_precision(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        F1_score = utils.get_F1_score(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        MCC = utils.get_MCC(TP = float(TP), FP = float(FP), FN = float(FN), TN = float(TN))
        print("accuracy: {}, recall: {}, precision: {}, F1_score: {}, MCC: {}".format(accuracy, recall, precision, F1_score, MCC))

else:
    raise Exception('mode error')
    
        
        





        
  


    
        
    
        
        





        
