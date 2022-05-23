import os
import sys
from turtle import width
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
    more_text = 'wrn_'
    Tot_Feats = 5
else:
    more_text = ''
    Tot_Feats = 6 # set to 6 to plot all and see INCLUDE below!
    if args.model_type == 'Stacked':
        epsilonz = np.array(list(range(0, 14, 2)))
    else:
        epsilonz = np.sort(np.array([1,3,5]+list(range(0, 14, 2))))

CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 

        'frost', 'fog', 'brightness', 'contrast', 
        'elastic_transform', 'pixelate', 'jpeg_compression', 

        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

extra_text = ''
if args.no_bn:
    extra_text = '_no_bn'
    Tot_Feats = 5 

all_means = {}
all_modes = {}
all_medians = {}
all_stds = {}
all_n_zeros = {}
all_n_poss = {}
all_n_negs = {}
for corruption in CORRUPTIONS:
    # print(corruption)
    with open(f'nps_activations{extra_text}/{more_text}means_{corruption}.pkl', 'rb') as handle:
        all_means[corruption] = pickle.load(handle)

    with open(f'nps_activations{extra_text}/{more_text}modes_{corruption}.pkl', 'rb') as handle:
        all_modes[corruption] = pickle.load(handle)

    with open(f'nps_activations{extra_text}/{more_text}medians_{corruption}.pkl', 'rb') as handle:
        all_medians[corruption] = pickle.load(handle)

    with open(f'nps_activations{extra_text}/{more_text}stds_{corruption}.pkl', 'rb') as handle:
        all_stds[corruption] = pickle.load(handle)


    with open(f'nps_activations{extra_text}/{more_text}n_zeros_{corruption}.pkl', 'rb') as handle:
        all_n_zeros[corruption] = pickle.load(handle)
    with open(f'nps_activations{extra_text}/{more_text}n_poss_{corruption}.pkl', 'rb') as handle:
        all_n_poss[corruption] = pickle.load(handle)
    with open(f'nps_activations{extra_text}/{more_text}n_negs_{corruption}.pkl', 'rb') as handle:
        all_n_negs[corruption] = pickle.load(handle)

def corr_to_feat_j(dict_input, j):
    eps_ordered_list_of_tuples = []
    for eps in epsilonz:
        eps_ordered_list_of_tuples.append(dict_input[eps])
    return np.array([tup[j] for tup in eps_ordered_list_of_tuples])

def convert_float_list_to_txt_list(list_of_epsilons):
    return [f'{eps}-robust model' for eps in list_of_epsilons]


# for Only_Feat_J in range(Tot_Feats):
#     for corruption, means in all_means.items():
#         os.makedirs(f'./plots_fires_corruptions', exist_ok=True)
#         plt.figure(figsize=(6,5))

#         J = Only_Feat_J
#         # print(corruption, corr_to_feat_j(means, J))
#         if corruption == 0:
#             plt.bar(epsilonz, corr_to_feat_j(means, J), color = 'g', edgecolor = 'black')#, label=f'Feat {J}', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=corr_to_feat_j(list(all_stds[corruption], 0), 
#         else:
#             plt.bar(epsilonz, corr_to_feat_j(means, J), color = 'g', edgecolor = 'black')#, label=f'Feat {J}', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=corr_to_feat_j(list(all_stds[corruption], 0), 
#         plt.xticks(epsilonz, convert_float_list_to_txt_list(means), rotation = 15)
        
#         # plt.legend()
#         if J == 0:
#             plt.ylabel(f'Mean of first Pre-ReLU & Pre-BN activations on {corruption}')
#         elif J == 1:
#             plt.ylabel(f'Mean of first Pre-ReLU activations on {corruption}')
#         elif J == 5:  
#             plt.ylabel(f'Mean of last Pre-ReLU activations on {corruption}')
#         else:
#             plt.ylabel(f'Mean of {J}th Pre-ReLU activations on {corruption}')
#         # plt.xlabel('$\epsilon$ (of $\epsilon$-robust model)')
#         plt.savefig(f'./plots_fires_corruptions/means{extra_text}_{corruption}_only_Feat_{J}.png')
#     # print(f'\n')

W = 0.2 # bar width! # INCLUDE width = W in plt.bar statements below!

# for corruption, means in all_means.items():
#     os.makedirs(f'./plots_fires_corruptions', exist_ok=True)
#     plt.figure(figsize=(6,5))
#     for J in range(Tot_Feats): # we have 5 features
#         # print(corruption, corr_to_feat_j(means, J))
#         if corruption == 0:
#             plt.bar(epsilonz+(J-Tot_Feats+1)*W, corr_to_feat_j(means, J), edgecolor = 'black', label=f'Feat {J}', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=corr_to_feat_j(list(all_stds[corruption], 0), 
#         else:
#             plt.bar(epsilonz+(J-Tot_Feats+1)*W, corr_to_feat_j(means, J), edgecolor = 'black', label=f'Feat {J}', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=corr_to_feat_j(list(all_stds[corruption], 0), 
#         plt.xticks(epsilonz, convert_float_list_to_txt_list(epsilonz), rotation = 15)
#     plt.legend()
#     plt.ylabel(f'Mean of first Pre-ReLU activation on {corruption}')
#     # plt.xlabel('$\epsilon$ (of $\epsilon$-robust model)')
#     plt.savefig(f'./plots_fires_corruptions/means{extra_text}_{corruption}.png')
# print(f'\n')

# for corruption, medians in all_medians.items():
#     plt.figure(figsize=(6,5))
#     for J in range(Tot_Feats): # we have 5 features
#         # print(corruption, corr_to_feat_j(medians, J))
#         if corruption == 0:
#             plt.bar(epsilonz+(J-Tot_Feats+1)*W, corr_to_feat_j(medians, J), color = 'g', edgecolor = 'black', label=f'Feat {J}', width = W, yerr=corr_to_feat_j(all_stds[corruption], 0), align='center', alpha=0.5, ecolor='black', capsize=10)
#         else:
#             plt.bar(epsilonz+(J-Tot_Feats+1)*W, corr_to_feat_j(medians, J), color = 'g', edgecolor = 'black', label=f'Feat {J}', width = W, yerr=corr_to_feat_j(all_stds[corruption], 0), align='center', alpha=0.5, ecolor='black', capsize=10)
#         plt.xticks(epsilonz, convert_float_list_to_txt_list(epsilonz), rotation = 15)
#     plt.legend()
#     plt.ylabel('Median of first Pre-ReLU activations')
#     # plt.xlabel('$\epsilon$ (of $\epsilon$-robust model)')
#     plt.savefig(f'./plots_fires_corruptions/medians{extra_text}_{corruption}.png')

    
# W = 0.2
# for corruption in CORRUPTIONS:
#     os.makedirs(f'./plots_fires_corruptions', exist_ok=True)
#     for J in range(Tot_Feats):
#         plt.figure(figsize=(6,5))
#         # print(corruption, corr_to_feat_j(all_n_zeros[corruption], J))
#         # print(corruption, corr_to_feat_j(all_n_poss[corruption], J))
#         # print(corruption, corr_to_feat_j(all_n_negs[corruption], J))
#         plt.bar(epsilonz+(-1)*W, corr_to_feat_j(all_n_zeros[corruption], J), edgecolor = 'black', label=f'num zeros', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) 
#         plt.bar(epsilonz+(0)*W, corr_to_feat_j(all_n_poss[corruption], J), edgecolor = 'black', label=f'num $>$0', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) 
#         plt.bar(epsilonz+(1)*W, corr_to_feat_j(all_n_negs[corruption], J), edgecolor = 'black', label=f'num $<$0', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) 
#         plt.xticks(epsilonz, convert_float_list_to_txt_list(epsilonz), rotation = 15)
#         plt.legend()
#         plt.ylabel(f'Pre-ReLU Activation {J} on {corruption}')
#         plt.savefig(f'./plots_fires_corruptions/nums{extra_text}_{corruption}_Feat{J}.png')

#         plt.figure(figsize=(6,5))
#         plt.bar(epsilonz+(0)*W, corr_to_feat_j(all_n_poss[corruption], J) - corr_to_feat_j(all_n_negs[corruption], J), edgecolor = 'black', label=f'num pos. - num neg.', width = W, align='center', alpha=0.5, ecolor='black', capsize=10) 
#         plt.xticks(epsilonz, convert_float_list_to_txt_list(epsilonz), rotation = 15)
#         plt.legend()
#         plt.ylabel(f'Pre-ReLU Activation {J} on {corruption}')
#         plt.savefig(f'./plots_fires_corruptions/nums_diff_{extra_text}_{corruption}_Feat{J}.png')


# Avg mean and pos,neg dist. across all corruptions
def get_avg_across_corruptions_for_feat_j(all_things, j):
    sum_of_np_arrays = 0
    for corruption in CORRUPTIONS:
        sum_of_np_arrays += corr_to_feat_j(all_things[corruption], j)
    return sum_of_np_arrays/len(CORRUPTIONS)
        
os.makedirs(f'./plots', exist_ok=True)
def get_equally_spaced_locations_for_epsilonz():
    return np.array(list(range(0, 2*len(epsilonz), 2)))

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

for J in range(Tot_Feats):
    equally_spaced_epsilonz = get_equally_spaced_locations_for_epsilonz()
    avg_n_poss_for_feat_J = get_avg_across_corruptions_for_feat_j(all_n_poss, J)
    avg_n_negs_for_feat_J = get_avg_across_corruptions_for_feat_j(all_n_negs, J)
    avg_means_for_feat_J = get_avg_across_corruptions_for_feat_j(all_means, J)

    # pos,neg plot
    fig, ax = plt.subplots(figsize=(6,5))
    ax.bar(equally_spaced_epsilonz, avg_n_negs_for_feat_J - avg_n_poss_for_feat_J, edgecolor = 'black', color = 'black', align='center', alpha=0.5, width = 1.75, ecolor='black', capsize=10) 
    ax.set_xticks(equally_spaced_epsilonz, epsilonz)#, rotation = 0
    if (J == Tot_Feats-2):
        extra_ylabel_text = f'Final Pre-ReLU Feature'
    elif (J == Tot_Feats-3):
        extra_ylabel_text = f'Penultimate Pre-ReLU Feature'
    else:
        extra_ylabel_text =  f'Pre-ReLU Feature {J}'
    ax.set_ylabel(f'Num. Negative - Num. Positive Values of {extra_ylabel_text}', fontsize= 20)
    ax.set_xlabel(f'$\epsilon$', fontsize= 20)
    ax.yaxis.set_major_formatter(human_format)
    ax.xaxis.set_tick_params(labelsize = 16)
    ax.yaxis.set_tick_params(labelsize = 16)
    plt.savefig(f'./plots/{more_text}nums_diff_{extra_text}_{corruption}_Feat_{J}.png', bbox_inches='tight')

    # mean plot
    fig, ax = plt.subplots(figsize=(8,7))
    ax.bar(equally_spaced_epsilonz, avg_means_for_feat_J, edgecolor = 'black', align='center', alpha=0.5, color = 'black', width = 1.75, ecolor='black', capsize=10) 
    ax.set_xticks(equally_spaced_epsilonz, epsilonz)#, rotation = 0
    ax.set_xlabel(f'$\epsilon$', fontsize= 20)
    ax.set_ylabel(f'Mean of {extra_ylabel_text}', fontsize= 20)
    ax.set_title(f'ResNet18', fontsize=20)
    ax.xaxis.set_tick_params(labelsize = 16)
    ax.yaxis.set_tick_params(labelsize = 16)
    plt.savefig(f'./plots/{more_text}means_{extra_text}_{corruption}_Feat_{J}.png', bbox_inches='tight')