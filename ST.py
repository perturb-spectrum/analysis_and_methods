'''
CUDA_VISIBLE_DEVICES=1 nohup python -u ST.py > logs/ST.log &
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet18_preReLU
from utils import *
from tqdm import tqdm
import argparse

# Hyperparams
parser = argparse.ArgumentParser(description='PyTorch CIFAR ST')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18 or WRN_28_10 or WRN_70_16')
parser.add_argument('--batch-size', default=512, type=int, help='batch size')
parser.add_argument('--num-epochs', default=20, type=int, help='num epochs')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
args = parser.parse_args()

save_epochs = 1

# SEED, VER, GPU
torch.manual_seed(0)
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
cifar_train = dsets.CIFAR10(root='../data/',
                          train=True,
                          transform=transform_train,
                          download=True)

cifar_val = dsets.CIFAR10(root='../data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

train_loader  = torch.utils.data.DataLoader(dataset=cifar_train,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=cifar_val,
                                         batch_size=args.batch_size,
                                         shuffle=False)

# DEFINE MODEL
if args.model_name == 'resnet18':
    model = ResNet18_preReLU()
elif args.model_name == 'WRN_70_16':
    model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
elif args.model_name == 'wrn':
    model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=2e-4) # optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 100], gamma=0.1)

# TRAIN MODEL 
for epoch in range(1, args.num_epochs+1):
    model.train()
    total_batch = len(cifar_train) // args.batch_size
    
    loader = tqdm(train_loader, dynamic_ncols=True) if not args.tqdm_off else train_loader
    for i, (batch_images, batch_labels) in enumerate(loader):
        X = batch_images.to(device)
        Y = batch_labels.to(device)

        pre = model(X)
        loss = criterion(pre, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if (epoch % save_epochs == 0) or (epoch == args.num_epochs):
        val_acc = clean_eval(model, val_loader, device, args)
        print(f'Epoch [{epoch}/{args.num_epochs}], Loss: {round(loss.item(), 4)}, Val accuracy: {round(val_acc, 4)}')
        os.makedirs(f'saved_state_dicts/ST_{args.model_name}', exist_ok=True)
        torch.save(model.state_dict(), f'saved_state_dicts/ST_{args.model_name}/ep_{epoch}.pth')
    else:
        print(f'Epoch [{epoch}/{args.num_epochs}], Loss: {round(loss.item(), 4)}')