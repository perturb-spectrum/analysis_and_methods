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
from models.dm_wide_resnet_for_pre_swish_feat import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours_no_bn import ResNet18_preReLU as  ResNet18_preReLU_no_bn
from models.resnet_ours import ResNet50, ResNet18_preReLU

from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGDAT')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18 preferred')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--batch-size', default=200, type=int, help='batch size')
parser.add_argument('--corruption', default='gaussian_noise', type=str, help='one of the CIFAR-10-C corruptions')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
parser.add_argument('--no-bn', default=False, action='store_true', help='set to eval no_bn models')
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

# DEFINE MODELs
models = {}
if args.model_name == 'wrn':
    epsilonz = list(range(0, 14, 2))
else:
    if args.model_type == 'Stacked':
        epsilonz = list(range(0, 14, 2))
    else:
        epsilonz = [1,3,5] + list(range(0, 14, 2))

print(f'==>For corruption {args.corruption} from test CIFAR-10-C dataset,')
extra_text = ''
if args.no_bn:
    extra_text = '_no_bn'
means = {}
modes = {}
medians = {}
stds = {}
n_zeros = {}
n_poss = {}
n_negs = {}
for epsilon in epsilonz:
    epsilon = float(epsilon)
    if args.model_name == 'resnet18':
        if args.no_bn:
            model = ResNet18_preReLU_no_bn()
        else:
            model = ResNet18_preReLU()

        if epsilon == 0:
            loc = f'./saved_state_dicts/ST{extra_text}_resnet18/ep_19.pth'
            model.load_state_dict(torch.load(loc), strict=True)
        else:
            loc = f'./saved_state_dicts/{args.model_type}PGDAT{extra_text}_resnet18_{epsilon}/ep_76.pth'
            model.load_state_dict(torch.load(loc), strict=True)

        print(f'loaded model at {loc}')
        model = model.to(device)

        (mean_0_pre_bn, mode_0_pre_bn, median_0_pre_bn, std_0_pre_bn, 
        mean_0, mode_0, median_0, std_0, 
        mean_1, mode_1, median_1, std_1, 
        mean_2, mode_2, median_2, std_2, 
        mean_3, mode_3, median_3, std_3, 
        mean_4, mode_4, median_4, std_4,
        n_zero_0_pre_bn, n_pos_0_pre_bn, n_neg_0_pre_bn,
        n_zero_0, n_pos_0, n_neg_0,
        n_zero_1, n_pos_1, n_neg_1,
        n_zero_2, n_pos_2, n_neg_2,
        n_zero_3, n_pos_3, n_neg_3,
        n_zero_4, n_pos_4, n_neg_4) = get_means_and_nums_of_pre_relu_activations_on_corruptions(model, cifar_val, device, args.corruption, args, f'model-trained-at-{epsilon}by255')
        
        means[epsilon] = (mean_0_pre_bn, mean_0, mean_1, mean_2, mean_3, mean_4)
        modes[epsilon] = (mode_0_pre_bn, mode_0, mode_1, mode_2, mode_3, mode_4)
        medians[epsilon] = (median_0_pre_bn, median_0, median_1, median_2, median_3, median_4)
        stds[epsilon] = (std_0_pre_bn, std_0, std_1, std_2, std_3, std_4)
        
        n_zeros[epsilon] = (n_zero_0_pre_bn, n_zero_0, n_zero_1, n_zero_2, n_zero_3, n_zero_4)
        n_poss[epsilon] = (n_pos_0_pre_bn, n_pos_0, n_pos_1, n_pos_2, n_pos_3, n_pos_4)
        n_negs[epsilon] = (n_neg_0_pre_bn, n_neg_0, n_neg_1, n_neg_2, n_neg_3, n_neg_4)

    elif args.model_name == 'wrn':
        model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        if epsilon == 0:
            loc = f'./saved_state_dicts/ST_wrn/ep_18.pth'
            model.load_state_dict(torch.load(loc), strict=True)
        else:
            loc = f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{epsilon}/ep_80.pth'
            model.load_state_dict(torch.load(loc), strict=True)
    
        print(f'loaded model at {loc}')
        model = model.to(device)

        (mean_0, mode_0, median_0, std_0, 
        mean_1, mode_1, median_1, std_1, 
        mean_2, mode_2, median_2, std_2, 
        mean_3, mode_3, median_3, std_3, 
        mean_4, mode_4, median_4, std_4,
        n_zero_0, n_pos_0, n_neg_0,
        n_zero_1, n_pos_1, n_neg_1,
        n_zero_2, n_pos_2, n_neg_2,
        n_zero_3, n_pos_3, n_neg_3,
        n_zero_4, n_pos_4, n_neg_4) = get_means_and_nums_of_pre_relu_activations_on_corruptions_wrn(model, cifar_val, device, args.corruption, args, f'model-trained-at-{epsilon}by255')
        
        means[epsilon] = (mean_0, mean_1, mean_2, mean_3, mean_4)
        modes[epsilon] = (mode_0, mode_1, mode_2, mode_3, mode_4)
        medians[epsilon] = (median_0, median_1, median_2, median_3, median_4)
        stds[epsilon] = (std_0, std_1, std_2, std_3, std_4)
        
        n_zeros[epsilon] = (n_zero_0, n_zero_1, n_zero_2, n_zero_3, n_zero_4)
        n_poss[epsilon] = (n_pos_0, n_pos_1, n_pos_2, n_pos_3, n_pos_4)
        n_negs[epsilon] = (n_neg_0, n_neg_1, n_neg_2, n_neg_3, n_neg_4)


os.makedirs(f'nps_activations{extra_text}', exist_ok=True)
if args.model_name == 'resnet18':
    with open(f'nps_activations{extra_text}/means_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(means, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/modes_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(modes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/medians_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(medians, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/stds_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(stds, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f'nps_activations{extra_text}/n_zeros_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_zeros, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/n_poss_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_poss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/n_negs_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_negs, handle, protocol=pickle.HIGHEST_PROTOCOL)



elif args.model_name == 'wrn':
    with open(f'nps_activations{extra_text}/wrn_means_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(means, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/wrn_modes_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(modes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/wrn_medians_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(medians, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/wrn_stds_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(stds, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f'nps_activations{extra_text}/wrn_n_zeros_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_zeros, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/wrn_n_poss_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_poss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'nps_activations{extra_text}/wrn_n_negs_{args.corruption}.pkl', 'wb') as handle:
        pickle.dump(n_negs, handle, protocol=pickle.HIGHEST_PROTOCOL)


        

