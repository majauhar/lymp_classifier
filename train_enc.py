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
from dataloader import AllDataset
from models.Autoencoder import Network

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test_enc(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, _, _, _, _ in dataloader:
            images = images.to(device)
            _, recons = model(images)
            loss = 100 * criterion(images, recons)
            
            total_loss += loss.item()
        
        print("Test Loss: {}".format(total_loss / len(dataloader)))
        

def train_enc(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, _, _, _, _ in train_loader:
            images = images.to(device)
            _, recons = model(images)
#             print(recons.shape)
            loss = 100 * criterion(images, recons)
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Train Loss: {}".format(epoch_loss / len(train_loader)))
        
#         test_enc(model, train_loader, criterion)
        test_enc(model, val_loader, criterion)
        
        if epoch == 49:
            PATH = '/kaggle/working/enc50.pth'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
        if epoch == 99:
            PATH = '/kaggle/working/enc100.pth'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
#         scheduler.step()
            
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(5)
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

enc_dataset = AllDataset(img_dir='/kaggle/input/classification-dataset/dlmi-lymphocytosis-classification/trainset',
                        annotations='/kaggle/input/classification-dataset/dlmi-lymphocytosis-classification/clinical_annotation.csv',
                        transform=transform)

VAL_SPLIT_RATIO = 0.2

enc_dataset_size = len(enc_dataset)
enc_dataset_indices = list(range(enc_dataset_size))

val_split_index = int(np.floor(VAL_SPLIT_RATIO * enc_dataset_size))

train_idx, val_idx = enc_dataset_indices[val_split_index:], enc_dataset_indices[:val_split_index]

enc_train_sampler = SubsetRandomSampler(train_idx)
enc_val_sampler = SubsetRandomSampler(val_idx)

enc_trainloader = DataLoader(dataset=enc_dataset, batch_size=128, sampler=enc_train_sampler, num_workers=4)

enc_valloader = DataLoader(dataset=enc_dataset, batch_size=128, sampler=enc_val_sampler, num_workers=4)

enc_model = Network()

enc_model = enc_model.to(device)

enc_criterion = nn.MSELoss()
enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=1e-2) # , betas=(0.9, 0.999), weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 0)
enc_scheduler = torch.optim.lr_scheduler.MultiStepLR(enc_optimizer, milestones=[10,20,40], gamma=0.1)

train_enc(enc_model, enc_trainloader, enc_valloader, 100, enc_criterion, enc_optimizer, enc_scheduler)