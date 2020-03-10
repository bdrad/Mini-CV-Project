#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import dataset
import random
import shutil
import model as m
import numpy as np
import pandas as pd

#TODO: change N to num of epochs
model_save_path = "end_model.pth"
def init_weights(model):
    model.fc.weight.data.fill_(0.01)
    model.fc2.weight.data.fill_(0.01)
    return model

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = ""
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'highest_accuracy_model.pth')

def get_roc_values(model, data_loader):
    rates = []
    for i in range(101):
        threshold = i / 100
        print("At threshold: " + str(threshold))
        total_predictions = 0
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        for x,y in data_loader:
            y = np.reshape(y, [y.shape[0], 2])
            y_hat = model(x)
            y_hat = f.normalize(y_hat, p=2, dim=1)
            print(y_hat)
            for i in range(y.shape[0]):
                _, max_index_y = y[i].max(0)
                if abs(y_hat[i][0])  > threshold:
                    #predicted positive case
                    if max_index_y.item() == 0:
                        true_positives += 1
                    else:
                        false_positives += 1
                elif max_index_y.item() == 0:
                    false_negatives += 1
                else:
                    true_negatives += 1
                total_predictions += 1
        print("tp: %3f" % true_positives)
        print("fp: %3f" % false_positives)
        tpr = 0 if true_positives == 0 else true_positives / (true_positives + false_negatives)
        fpr = 0 if false_positives == 0 else false_positives / (true_negatives + false_positives)
        rates.append([tpr,fpr])
    pd.DataFrame(rates).to_csv("roc_values.csv")


def get_metrics(model, data_loader):
    total_predictions = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    print("getting metrics...")
    for x,y in data_loader:
        y = np.reshape(y, [y.shape[0], 2])
        y_hat = model(x)
        for i in range(y.shape[0]):
            max_index_y = y[i].argmax()
            max_index_hat = y_hat[i].argmax()
            print("Max_y: %03d, Max_hat: %03d" % (max_index_y, max_index_hat))
            if max_index_y.item() == max_index_hat.item():
                if max_index_y.item() == 0:
                    true_positives += 1
                else:
                    true_negatives += 1
            elif max_index_hat.item() == 0:
                false_positives += 1
            else:
                false_negatives += 1
            total_predictions += 1
    accuracy = (true_positives + true_negatives) / total_predictions
    print("""True positive: %03d    | False positive: %03d
             False negative: %03d   | True negative: %03d
             Total predictions: %03d| Accuracy: %.4f
             """% (true_positives,
             false_positives, false_negatives, true_negatives,
             total_predictions, accuracy))
    return accuracy


def main():
    #For model
    #For images
    #import resnet 50 layers to fine tune
    model = m.resnet18(1, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    custom_transform = transforms.Compose([transforms.ToTensor()])
    #Load datasets
    print("Loading data...")
    train = dataset.CBISDataset("Train")
    test = dataset.CBISDataset("Test")

    train_dl = DataLoader(train,
                         batch_size=10,
                         shuffle=True,
                         num_workers=1)
    test_dl = DataLoader(test,
                        batch_size=10,
                        shuffle=True,
                        num_workers=1)
    loss = nn.MSELoss()
    #categorical cross entropy

    print("Data loaded")
    epochs = 10
    best_acc = -10000
    print("Beginning training...")
    
    for epoch in range(epochs):
        scheduler.step()
        model.train()
        for batch_idx, (x,y) in enumerate(train_dl):
             print("beg batch")
             y_hat = model(x)
             y = np.reshape(y, [y.shape[0], 2])
             cost = loss(y, y_hat)
             optimizer.zero_grad()

             #Update model parameters
             cost.backward()

             #Update model parameters
             optimizer.step()
             #Logging
             if not batch_idx % 10:
                 print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                        %(epoch+1, epochs, batch_idx,
                          len(train_dl), cost))
                 print("Lr: " + str(scheduler.get_lr()))
        #Begin inference
        model.eval()
        with torch.set_grad_enabled(False):
            # save memory during inference
            #input("About to start validation")
            acc = get_metrics(model, test_dl)

            # remember best acc and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_acc,
            }, is_best)
            print('Epoch: %03d/%03d | Train: %.3f%% | Test: %.3f%%' % (
                epoch+1, epochs,
                 get_metrics(model, train_dl),
                    acc))
    #Uncomment this line if you want the ROC values
    #get_roc_values(model, test_dl)

    #save model weights
    print("Saving model: "+ model_save_path + " locally")
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
