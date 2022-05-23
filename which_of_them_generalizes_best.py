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
from utils import *
import argparse
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet50, ResNet18_preReLU

from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGDAT')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18 preferred')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--batch-size', default=200, type=int, help='batch size')
parser.add_argument('--epsilon-test', default=2, type=float, help='perturbation budget = epsilon / 255')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
parser.add_argument('--loss-type', default='L1', type=str, help='L1 or L2 or ssim or ')
# parser.add_argument('--current-expt', default='test_check', type=str, help='test_check (original!) or train_check')
args = parser.parse_args()

save_epochs = 10

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

# DEFINE MODELs
models = {}
if args.model_name == 'wrn':
    epsilonz = list(range(0, 14, 2))
else:
    if args.model_type == 'Stacked':
        epsilonz = list(range(0, 14, 2))
        epochs = {2: 100, 4: 90, 6: 80, 8: 80, 10: 80, 12: 76}
    else:
        epsilonz = list(range(0, 13, 1))
        # epochs = {1: 120, 2: 170, 3: 76, 4: 76, 5: 76, 6: 76, 7: 76, 8: 76, 9: 76, 10: 76, 11: 76, 12: 76}

print(f'==>For PGD attack test of strength {args.epsilon_test}/255 on both train and test CIFAR dataset,')
for epsilon in epsilonz:
    epsilon = float(epsilon)
    if args.model_name == 'resnet18':
        model = ResNet18_preReLU()
        if epsilon == 0:
            model.load_state_dict(torch.load(f'./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
        else:
            model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{epsilon}/ep_76.pth'), strict=True)
            # model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{epsilon}/ep_{epochs[epsilon]}.pth'), strict=True)
    elif args.model_name == 'wrn':
        model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        if epsilon == 0:
            model.load_state_dict(torch.load(f'./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
        else:
            model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{epsilon}/ep_80.pth'), strict=True)
    model = model.to(device)


    if args.epsilon_test == 0:
        clean_acc_train = clean_eval(model, train_loader, device, args)
        print(f'Model {args.model_type}Trained with strength {epsilon}/255 has clean accs: TRAIN {clean_acc_train}')

        clean_acc_test = clean_eval(model, val_loader, device, args)
        print(f'Model {args.model_type}Trained with strength {epsilon}/255 has clean accs: TEST {clean_acc_test}')
    else:
        pgd_20_acc_train = pgd_eval_for_epsilon(model, train_loader, device, args.epsilon_test, args)
        print(f'Model {args.model_type}Trained with strength {epsilon}/255 has PGD Adv accs: TRAIN {pgd_20_acc_train}')

        pgd_20_acc_test = pgd_eval_for_epsilon(model, val_loader, device, args.epsilon_test, args)
        print(f'Model {args.model_type}Trained with strength {epsilon}/255 has PGD Adv accs: TEST {pgd_20_acc_test}')
        

        

