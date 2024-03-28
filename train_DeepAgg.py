import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image
from itertools import chain

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from PIL import Image

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support

from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import random
from copy import deepcopy
import torch.nn.functional as F

from collections import OrderedDict
import cv2

# local imports
from models.Autoencoder import CNN_Encoder_Resnet
from dataloader import BagDataset
from models.MILAgg import *

def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
        
def test(model, dataloader):
    bag_labels = []
    bag_predictions = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, label = data['images'], data['label']

            images = torch.cat(images, dim=0).to(device)
    #             label = label.to(device)

            classes, bag_prediction, _, _ = model(images)
#             bag_prediction = model(images)

            bag_labels.append(label.numpy())
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())

#     five_scores_bag_predictions = bag_predictions
    bag_predictions = [0 if prediction < 0.5 else 1 for prediction in bag_predictions]
#     print(bag_labels, bag_predictions)

    balanced_acc = balanced_accuracy_score(bag_labels, bag_predictions)
    normal_acc = accuracy_score(bag_labels, bag_predictions)
        
    # --- Printing evaluation numbers every 5 epochs ---
#         correct = 0
#         for i in range(len(bag_predictions)):
#             if bag_predictions[i] == bag_labels[i]:
#                 correct += 1
    print("Balanced Acc: {:.4f} Normal Acc: {:.4f}".format(balanced_acc, normal_acc))
    return balanced_acc
    
#     acc, auc_value, precision, recall, fscore = five_scores(bag_labels, five_scores_bag_predictions)
#     print(acc, auc_value, precision, recall, fscore)
    
        

feature_extractor = CNN_Encoder_Resnet(256)
local_checkpoint = torch.load('/kaggle/working/enc100.pth')['model_state_dict']

new_state_dict = OrderedDict()
count = 0
for k, v in local_checkpoint.items():
    count += 1
    name = k[8:] # remove `module.`
#     print(name)
    new_state_dict[name] = v
    

load_my_state_dict(feature_extractor, new_state_dict)
# print(count)

img_path = '/kaggle/input/classification-dataset/dlmi-lymphocytosis-classification/trainset'
train_df = pd.read_csv('/kaggle/input/classification-dataset/train_set.csv')
val_df = pd.read_csv('/kaggle/input/classification-dataset/val_set.csv')


train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(5)
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = BagDataset(img_path, train_df, train_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, shuffle=True)

val_dataset = BagDataset(img_path, val_df, val_transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)

num_feats = 256
i_classifier = FCLayer(num_feats, 1)
b_classifier = BClassifier(input_size=num_feats, output_class=1)
model = FullNet(i_classifier, b_classifier, feature_extractor)

# load pretrained model
# local_checkpoint = torch.load('/kaggle/working/frozen_enc_best.pth')
# model.load_state_dict(local_checkpoint['model_state_dict'])
# model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam([{'params': extractor_params, 'lr': 1e-5}, {'params': normal_params, 'lr': 1e-2}], betas=(0.5, 0.9), weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,100], gamma=0.1)


train_losses = []
val_losses = []
accuracies = []
bal_accs = []
roc_accs = []

best_acc = 0.0
for epoch in range(150):
    epoch_loss = 0.0
    model.train()
    print("-- training: epoch {}".format(epoch+1))
#     ground_labels = []
#     bag_predictions = []
#     max_predictions = []
    for data in train_loader:
        images, label = data['images'], data['label']
#         ground_labels.append(label)
        images = torch.cat(images, dim=0).to(device)
        label = label.to(device)


        classes, bag_prediction, _, _ = model(images) # n X L
        max_prediction, index = torch.max(classes, 0)
#         print(bag_prediction, max_prediction)
#         bag_predictions.append(bag_prediction)
#         max_predictions.append(max_prediction)
#         torch.cuda.empty_cache()

#         bag_predictions = torch.cat(bag_predictions, dim=0)
#         max_predictions = torch.cat(max_predictions, dim=0)
        loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1).float())
        loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1).float())
        
#         prediction = model(images)
#         loss = criterion(prediction.view(1, -1), label.view(1, -1).float())
        weight = 1.0
        if label.item() == 1:
#             print("label: {}".format(label.item()))
            weight_factor = 0.30
        else:
#             print("label: {}".format(label.item()))
            weight_factor = 0.70
        
        loss_total = (0.5*loss_bag + 0.5*loss_max) * weight_factor
#         loss_total = loss * weight_factor
#         print(type(loss_total), loss_total.shape)
        loss_total = loss_total.mean()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()  

        epoch_loss += loss_total.item()

    print("Epoch loss: {:.4f}".format(epoch_loss))
    train_losses.append(epoch_loss)
#     test_loss = 0.0
    

    # ------------ Testing ------------
    test(model, train_loader)
    val_acc = test(model, val_loader)
#     bag_labels = []
#     bag_predictions = []
#     model.eval()
#     with torch.no_grad():
#         for data in val_loader:
#             images, label = data['images'], data['label']

#             images = torch.cat(images, dim=0).to(device)
#     #             label = label.to(device)

#             classes, bag_prediction, _, _ = model(images)

#             bag_labels.append(label.numpy())
#             bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())

#     bag_predictions = [0 if prediction < 0.5 else 1 for prediction in bag_predictions]
# #     print(bag_labels, bag_predictions)

#     balanced_acc = balanced_accuracy_score(bag_labels, bag_predictions)
#     scikit_acc = accuracy_score(bag_labels, bag_predictions)
#     roc_acc = roc_auc_score(bag_labels, bag_predictions)

#     accuracies.append(scikit_acc)
#     bal_accs.append(balanced_acc)
#     roc_accs.append(roc_acc) 
        
#     # --- Printing evaluation numbers every 5 epochs ---
#     if (epoch+1) % 5 == 0:
#         print("-- testing: epoch {}".format(epoch+1))
# #         correct = 0
# #         for i in range(len(bag_predictions)):
# #             if bag_predictions[i] == bag_labels[i]:
# #                 correct += 1
#         print("Balanced Acc: {:.4f} scikit-acc: {:.4f} roc_score: {:.4f}".format(balanced_acc, scikit_acc, roc_acc))
    
    # --- Save the model every 50 epochs ---
    if (epoch+1) % 50 == 0:
        PATH = '/kaggle/working/frozen_enc_2' + str(epoch+1) + '.pth'
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)
    
    # --- Save the best model ---
    if val_acc > best_acc:
        best_acc = val_acc
        PATH = '/kaggle/working/frozen_enc_2best.pth'
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)


    scheduler.step()