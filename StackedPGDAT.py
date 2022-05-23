import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD
from utils import clean_eval, pgd_eval, get_timestamp
import argparse
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet50, ResNet18_preReLU

from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGDAT')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18 preferred')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--num-epochs', default=200, type=int, help='num epochs')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation budget = epsilon / 255')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
args = parser.parse_args()

save_epochs = 100

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
if args.model_name == 'resnet50':
    model = ResNet50()
if args.model_name == 'resnet18':
    model = ResNet18_preReLU()
    prev_level = args.epsilon-2.0
    if prev_level == 0:
        model.load_state_dict(torch.load(f'./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
    else:
        model.load_state_dict(torch.load(f'./saved_state_dicts/StackedPGDAT_resnet18_{prev_level}/ep_76.pth'), strict=True)
    # for name, param in model.named_parameters(): # Ref: https://discuss.pytorch.org/t/how-can-i-disable-all-layers-gradient-expect-the-last-layer-in-pytorch/53043
    #     print(name, param.requires_grad)
    # exit()
if args.model_name == 'wrn':
    model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
    prev_level = args.epsilon-2.0
    if prev_level == 0:
        model.load_state_dict(torch.load(f'./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
    else:
        model.load_state_dict(torch.load(f'./saved_state_dicts/StackedPGDAT_wrn_{prev_level}/ep_80.pth'), strict=True)
    
model = model.to(device)
criterion_ce = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=2e-4) # optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 100], gamma=0.1)

atk = PGD(model, eps=args.epsilon/255.0, alpha=0.007, steps=10, random_start=True) 

# TRAIN MODEL 
for epoch in range(1, args.num_epochs+1):
    model.train()
    total_batch = len(cifar_train) // args.batch_size
    
    loader = tqdm(train_loader, dynamic_ncols=True) if not args.tqdm_off else train_loader
    for (batch_images, batch_labels) in loader:
        X_nat = batch_images.to(device) # prev: had batch_images.clone().to(device) here
        X_adv = atk(batch_images, batch_labels).to(device)
        Y = batch_labels.to(device)

        logits_adv = model(X_adv)

        loss = criterion_ce(logits_adv, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if (epoch % save_epochs == 0) or (epoch == args.num_epochs) or (epoch == 76 and args.model_name == 'resnet18'): # 76 is args.num_epochs in TRADES github
        val_acc = clean_eval(model, val_loader, device, args)
        pgd20_adv_acc = pgd_eval(model, val_loader, device, args)
        print(f'[{get_timestamp()}] Epoch [{epoch}/{args.num_epochs}], Loss: {round(loss.item(), 4)}, Val accuracy: {round(val_acc, 4)}, PGD Adv accuracy: {round(pgd20_adv_acc, 4)}')
        
        os.makedirs('saved_state_dicts', exist_ok=True)
        os.makedirs(f'saved_state_dicts/StackedPGDAT_{args.model_name}_{args.epsilon}', exist_ok=True)
        torch.save(model.state_dict(), f'saved_state_dicts/StackedPGDAT_{args.model_name}_{args.epsilon}/ep_{epoch}.pth')
    else:
        val_acc = clean_eval(model, val_loader, device, args)
        print(f'[{get_timestamp()}] Epoch [{epoch}/{args.num_epochs}], Loss: {round(loss.item(), 4)}, Val accuracy: {round(val_acc, 4)}')
