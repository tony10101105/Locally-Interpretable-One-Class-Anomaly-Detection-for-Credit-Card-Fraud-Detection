import torch
import matplotlib.pyplot as plt

def plot_and_save_fig(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color = 'darkorange', lw = 1, label='AUC = %0.4f'% roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw = 1, linestyle='--')
    plt.legend(loc='lower right')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ion()
    plt.savefig('./fig/AUROC')
    plt.pause(2)
    plt.close()

def load_checkpoint(g_path, d_path):
    
    generator = models.autoencoder()
    discriminator = models.FCNN()

    g_checkpoint = torch.load(g_path)
    d_checkpoint = torch.load(d_path)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.0002, weight_decay = 1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, weight_decay = 1e-3)

    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
    d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])

    generator.load_state_dict(g_checkpoint['model_state_dict'])
    discriminator.load_state_dict(d_checkpoint['model_state_dict'])

    assert g_checkpoint['epoch'] == d_checkpoint['epoch'], 'epoch number loading error'
    current_epoch = g_checkpoint['epoch']

    return generator, discriminator, g_optimizer, d_optimizer, current_epoch

def get_MCC(TP, FP, FN, TN):
    try:
        MCC = ( TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** (1/2)
        MCC = round(MCC, 4)
    except:
        MCC = 0
    return MCC

def get_accuracy(TP, FP, FN, TN):
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    accuracy = round(accuracy, 4)
    return accuracy

def get_recall(TP, FP, FN, TN):
    try:
        recall = TP / (TP + FN)
        recall = round(recall, 4)
    except:
        recall = 'N/A'
    return recall

def get_precision(TP, FP, FN, TN):
    try:
        precision = TP / (TP + FP)
        precision = round(precision, 4)
    except:
        precision = 'N/A'
    return precision

def get_F1_score(TP, FP, FN, TN):
    try:
        recall = get_recall(TP = TP, FP = FP, FN = FN, TN = TN)
        precision = get_precision(TP = TP, FP = FP, FN = FN, TN = TN)
        F1_score = 2 / ((1 / recall) + (1 / precision))
        F1_score = round(F1_score, 4)
    except:
        F1_score = 'N/A'
    return F1_score
