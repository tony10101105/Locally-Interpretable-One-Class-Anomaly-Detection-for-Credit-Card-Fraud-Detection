import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
import models
import DataSet
import utils
import random
import lime
import lime.lime_tabular


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 4096)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--n_epochs", type = int, default = 2)
parser.add_argument("--normalization", type = str, default = 'z_score')
parser.add_argument("--reconstructionLoss", type = str, default = 'MSE')
parser.add_argument("--mode", type = str, default = 'explainer')
parser.add_argument("--GPU", type = bool, default = False)
parser.add_argument("--resume", type = bool, default = False)
args = parser.parse_args()
print(args)

torch.manual_seed(4)#for reproducibility
random.seed(0)

#load datasets
print('loading the dataset...')

non_fraud_Data = DataSet.SplitedDataSet(mode = 'non-fraud')
fraud_Data = DataSet.SplitedDataSet(mode = 'fraud')

data_point_num = len(non_fraud_Data)
test_data_point_num = 490
train_data_point_num = data_point_num - test_data_point_num
trainData, nonFraudTestData = random_split(non_fraud_Data, [train_data_point_num, test_data_point_num])
#trainData, _ = random_split(trainData, [700, len(trainData) - 700])
trainData = DataSet.DataSet([trainData], args.normalization)
fraud_Data, _ = random_split(fraud_Data, [490, 2])
testData = DataSet.DataSet([nonFraudTestData, fraud_Data], args.normalization) #following the setting of 13.pdf


trainDataLoader = DataLoader(dataset = trainData, batch_size = args.batch_size, shuffle = True, drop_last=True)
testDataLoader = DataLoader(dataset = testData, batch_size = args.batch_size, shuffle = True)
print('datasets loading finished!')

#load models
if args.resume or args.mode == 'test' or 'explainer' in args.mode:
    try:
        print('loading existing (pretrained) models...')
        g_path = './checkpoints/g_checkpoint.pth'
        d_path = './checkpoints/d_checkpoint.pth'
        generator, discriminator, g_optimizer, d_optimizer, current_epoch = utils.load_checkpoint(g_path, d_path)
    except:
        raise Exception('failed to load models. Please check the path')
else:
    print('building new models...')
    generator = models.autoencoder()
    discriminator = models.FCNN()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr = args.lr, weight_decay = 1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = args.lr, weight_decay = 1e-3)
    current_epoch = 0


if args.GPU == True and torch.cuda.is_available():
    print('using GPU...')
    generator = generator.cuda()
    discriminator = discriminator.cuda()

#some checks
if current_epoch >= args.n_epochs and args.mode == 'train':
    raise Exception('epoch number error!')
if args.mode == 'test' and (os.path.exists('./checkpoints/g_checkpoint.pth') and os.path.exists('./checkpoints/d_checkpoint.pth')) == False:
    raise Exception('no pretrained models for testing stage')

#setting the loss function
#setting re-construction loss
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

#if we use BCEWithLogitsLoss, sigmoid should be manually applied in testing stage
BCELoss = nn.BCEWithLogitsLoss()
sig = nn.Sigmoid()

#start training / testing
if args.mode == 'train':
    generator.train()
    discriminator.train()
    print('start running on train mode...')
    for epoch in range(current_epoch, args.n_epochs):
        print('epoch:', epoch + 1)
        #for recording losses and accuracy/MCC
        g_loss_Re =0
        g_loss_BCE =0
        d_loss_sum = 0
        for i, (features, labels) in enumerate(trainDataLoader):
            labels = labels.unsqueeze(1)
            if torch.sum(labels) != 0:
                raise Exception('stop')
                
            real_label = torch.ones(labels.size())
            fake_label = torch.zeros(labels.size())

            if args.GPU == True and torch.cuda.is_available():
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                features = features.cuda()
                labels = labels.cuda()

            ##train Generator
            reconstruction = generator(features)
            Re_Loss = reconstructionLoss(reconstruction, features)
            fake_pred = discriminator(reconstruction)
            
            BCE_Loss = BCELoss(fake_pred, real_label)
            g_loss = Re_Loss + BCE_Loss
            g_loss = Re_Loss
            
            g_optimizer.zero_grad()
            g_loss.backward()
        
            '''nn.utils.clip_grad_norm_(MIX.parameters(), args.clipping_value)'''
            g_optimizer.step()

            g_loss_Re += torch.sum(Re_Loss)
            g_loss_BCE += torch.sum(Re_Loss)
        
            ##train discriminator
            real_pred = discriminator(features)
            real_loss = BCELoss(real_pred, real_label)

            fake_pred = discriminator(reconstruction.detach())
            fake_loss = BCELoss(fake_pred, fake_label)

            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            '''nn.utils.clip_grad_norm_(GRU.parameters(), args.clipping_value)'''
            d_optimizer.step()

            d_loss_sum += torch.sum(d_loss)


            if (i + 1) % 10 == 0:
                print("iteration: {} / {}, Epoch: {} / {}, g_loss_Re: {:.5f}, g_loss_BCE: {:.4f}, d_loss: {:.4f}".format(
                    str((i+1)*args.batch_size), str(train_data_point_num), epoch+1, args.n_epochs, g_loss_Re.data / (10*args.batch_size), g_loss_BCE.data / (10*args.batch_size), d_loss_sum.data / (10*args.batch_size)))
                g_loss_Re = 0
                g_loss_BCE = 0
                d_loss_sum = 0
                print('real_pred:', real_pred)
                print('fake_pred:', fake_pred)

        torch.save({'epoch': epoch+1, 'model_state_dict': generator.state_dict(), 'optimizer_state_dict': g_optimizer.state_dict()}, './checkpoints/g_checkpoint.pth')
        torch.save({'epoch': epoch+1, 'model_state_dict': discriminator.state_dict(), 'optimizer_state_dict': d_optimizer.state_dict()}, './checkpoints/d_checkpoint.pth')


elif args.mode == 'test':
    generator.eval()
    discriminator.eval()
    print('start running on test mode...')
    for i in range(1, 100):
        TP, FP, TN, FN = 0, 0, 0, 0
        all_pred = []
        all_labels = []
        args.threshold = i / 100
        print('threshold:', args.threshold)
        for i, (features, labels) in enumerate(testDataLoader):

            if args.GPU == True and torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()

            ##test Discriminator
            reconstruction = generator(features)
            p_fraud = discriminator(reconstruction)
            p_fraud = sig(p_fraud)
            #print('re:', torch.sum(features - reconstruction, 1))
            p_fraud = p_fraud.squeeze()
            #p_fraud = 1 - p_fraud
            #print('p_fraud:', p_fraud)
            #print('labels:', labels)

            all_pred.extend(p_fraud.tolist())
            all_labels.extend(labels.tolist())
            
            final_pred = torch.zeros(p_fraud.size())
            final_pred[p_fraud >= args.threshold] = 1
        
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
        
elif args.mode == 'explainer':
    def AE_prediction(features, model=generator):
        model.eval()
        features = torch.from_numpy(features).float()
        reconstruction = model(features)
        Re_loss_pred = (torch.sum(features - reconstruction, 1)**2) / 28
        return Re_loss_pred.detach().cpu().numpy()

    def D_prediction(features, model=discriminator):
        #features is already the output of AE, namely the reconstruction
        model.eval()
        reconstruction = torch.from_numpy(features).float()
        p_fraud = model(reconstruction)
        p_fraud = sig(p_fraud)
        p_genuine = 1 - p_fraud
        return torch.cat([p_fraud, p_genuine], 1).detach().cpu().numpy()
    
    def AED_prediction(features, model=[generator, discriminator]):
        AE, discriminator = model[0], model[1]
        AE.eval()
        discriminator.eval()
        features = torch.from_numpy(features).float()
        reconstruction = AE(features)
        p_fraud = discriminator(reconstruction)
        p_fraud = sig(p_fraud)
        p_genuine = 1 - p_fraud
        return torch.cat([p_fraud, p_genuine], 1).detach().cpu().numpy()

    features = np.array([i[0].numpy() for i in testData])
    reconstructed_features = generator(torch.from_numpy(features).float()).detach().numpy()
    labels = np.array([i[1].numpy() for i in testData])
    f_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    print('features[-5]:', features[-5])
    #explain Generator(AutoEncoder)
    AE_explainer = lime.lime_tabular.LimeTabularExplainer(features,
                                                mode='regression',
                                                feature_names=f_names,
                                                verbose=True,
                                                class_names=['reconstruction error'])
    AE_exp = AE_explainer.explain_instance(features[-5],
                                    AE_prediction,
                                    num_features=6)
    AE_exp.save_to_file('AE_lime.html')
    print('AE explaination done')

    #explain Discriminator
    D_explainer = lime.lime_tabular.LimeTabularExplainer(reconstructed_features,
                                                mode='classification',
                                                feature_names=f_names,
                                                verbose=True,
                                                class_names=['Fraud', 'Genuine'])
    D_exp = D_explainer.explain_instance(reconstructed_features[-5],
                                    D_prediction,
                                    num_features=6)
    D_exp.save_to_file('D_lime.html')
    print('Discriminator explaination done')

    #explain whole network
    AED_explainer = lime.lime_tabular.LimeTabularExplainer(features,
                                                mode='classification',
                                                feature_names=f_names,
                                                verbose=True,
                                                class_names=['Fraud', 'Genuine'])
    AED_exp = AED_explainer.explain_instance(features[-5],
                                    AED_prediction,
                                    num_features=6)
    AED_exp.save_to_file('AED_lime.html')
    print('whole network explaination done')
        
    
        
        





        
  


    
        
    
        
        





        
