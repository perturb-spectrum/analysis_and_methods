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
from utils import *
import argparse
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet18_preReLU
from models.ifade_net import ODEf, Stable, ODEf_smaller
import copy, time

from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle
import time

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch CIFAR IFaDe_after_PGDAT')
parser.add_argument('--data-dir', default='../data', type=str, help='directory to store cifar data')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18')
parser.add_argument('--batch-size', default=200, type=int, help='batch size')
parser.add_argument('--num-epochs', default=200, type=int, help='num epochs')
parser.add_argument('--scale', default=1.0, type=float, help='scaling factor for scaled floor quantization')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
parser.add_argument('--epsilon-test', default=8, type=float, help='perturbation budget = epsilon / 255')
args = parser.parse_args()

save_epochs = 1

# SEED, VER, GPU
torch.manual_seed(0)
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
# device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
cifar_train = dsets.CIFAR10(root=f'{args.data_dir}/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

cifar_val = dsets.CIFAR10(root=f'{args.data_dir}/',
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

# DEFINE PRETRAINED MODEL and LOAD
if args.model_name == 'wrn':
    epsilonz = list(range(0, 14, 2))
else:
    if args.model_type == 'Stacked':
        epsilonz = list(range(0, 14, 2))
    else:
        epsilonz = list(range(0, 13, 1))

for epsilon in epsilonz:
    epsilon = float(epsilon)
    args.epsilon_test = float(args.epsilon_test)
    if args.model_name == 'resnet18':
        model = ResNet18_preReLU()
        if epsilon == 0:
            model.load_state_dict(torch.load(f'./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
        else:
            model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{epsilon}/ep_76.pth'), strict=True)
        model = model.to(device)
        many_feature_full_quant_eval_BPDA(model, val_loader, args.epsilon_test, device, args.data_dir, f'AT_{epsilon}_{args.model_name}_eval_on_{args.epsilon_test}', args, args.scale)
    
    elif args.model_name == 'wrn':
        model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        if epsilon == 0:
            model.load_state_dict(torch.load(f'./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
        else:
            model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{epsilon}/ep_80.pth'), strict=True)
        model = model.to(device)
        many_feature_full_quant_eval_generic_BPDA(model, val_loader, args.epsilon_test, device, args.data_dir, f'AT_{epsilon}_{args.model_name}_eval_on_{args.epsilon_test}', args, args.scale)

