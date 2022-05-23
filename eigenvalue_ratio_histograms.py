"""
AT and Feature Quantization
"""

import os
import numpy as np
import torch
import torchvision.utils
import torchattacks
from utils import *
import argparse
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet18_preReLU

from utils import *

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', default='resnet_18', type=str, help='dm_wrn_28_10 or resnet_18')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
args = parser.parse_args()


# SEED, VER, GPU
torch.manual_seed(0)
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
device = torch.device("cpu") 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINE PRETRAINED MODEL and LOAD
if args.model == 'resnet_18':
    pretrained_nr_model = ResNet18_preReLU()
    pretrained_nr_model.load_state_dict(torch.load('./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
    pretrained_nr_model = pretrained_nr_model.to(device)
    os.makedirs(f'./plots', exist_ok=True)

    count = 0
    epsilonz = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] if args.model_type == 'Stacked' else [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    for eps in epsilonz:
        count += 1

        pretrained_r_model = ResNet18_preReLU() 
        pretrained_r_model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{eps}/ep_76.pth'), strict=True)
        pretrained_r_model = pretrained_r_model.to(device)

        print(f'conv1: ')
        if count == 1: print(f'non-robust', penalty_ratio(pretrained_nr_model.conv1.weight, device, f'non-{args.model_type}-robust-first'))
        print(f'robust', penalty_ratio(pretrained_r_model.conv1.weight, device, f'{eps}-{args.model_type}-robust-first'))

        print(f'\nlayer4[1].conv2: ')
        if count == 1: print(f'non-robust', penalty_ratio(pretrained_nr_model.layer4[1].conv2.weight, device, f'non-{args.model_type}-robust-last'))
        print(f'robust', penalty_ratio(pretrained_r_model.layer4[1].conv2.weight, device, f'{eps}-{args.model_type}-robust-last'))

elif args.model == 'wrn':
    pretrained_nr_model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
    
    pretrained_nr_model.load_state_dict(torch.load('./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
    pretrained_nr_model = pretrained_nr_model.to(device)
    os.makedirs(f'./plots', exist_ok=True)

    count = 0
    for eps in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        count += 1

        pretrained_r_model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        pretrained_r_model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{eps}/ep_80.pth'), strict=True)
        pretrained_r_model = pretrained_r_model.to(device)

        print(f'init_conv: ')
        if count == 1: print(f'non-robust', penalty_ratio_wrn(pretrained_nr_model.init_conv.weight, device, f'non-{args.model_type}-robust-first'))
        print(f'robust', penalty_ratio_wrn(pretrained_r_model.init_conv.weight, device, f'{eps}-{args.model_type}-robust-first'))

        print(f'\nlayer[2].block[3]: ')
        if count == 1: print(f'non-robust', penalty_ratio_wrn(pretrained_nr_model.layer[2].block[3].conv_1.weight, device, f'non-{args.model_type}-robust-last'))
        print(f'robust', penalty_ratio_wrn(pretrained_r_model.layer[2].block[3].conv_1.weight, device, f'{eps}-{args.model_type}-robust-last'))

    
