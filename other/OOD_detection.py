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
from models.resnet_ours import ResNet50, ResNet18_preReLU

from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGDAT')
parser.add_argument('--model-name', default='resnet18', type=str, help='resnet18 preferred')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--batch-size', default=200, type=int, help='batch size')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
args = parser.parse_args()

save_epochs = 10

# SEED, VER, GPU
torch.manual_seed(0)
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINE MODELs
models = {}
if args.model_type == 'Stacked':
    epsilonz = list(range(0, 14, 2))
    deltaz = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
else:
    epsilonz = list(range(0, 13, 1))
    deltaz = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


for epsilon in epsilonz:
    for epsilon_test in deltaz:
        epsilon = float(epsilon)
        epsilon_test = float(epsilon_test)
        
        if epsilon < epsilon_test:
            print(f'{epsilon}-robust-model evaluated on {epsilon_test}')
            if args.model_name == 'resnet18':
                model = ResNet18_preReLU()
                if epsilon == 0:
                    model.load_state_dict(torch.load(f'./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
                else:
                    model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{epsilon}/ep_76.pth'), strict=True)
            model = model.to(device)

            scores = torch.load(f'./scores/{epsilon}_resnet18_eval_on_{epsilon_test}.pth')
            scores_of_iD_data, scores_of_OOD_data = scores['scores_of_iD_data'], scores['scores_of_OOD_data']

            GTs = []
            GTs.extend( [1 for _ in range(len(scores_of_iD_data))] )
            GTs.extend( [0 for _ in range(len(scores_of_OOD_data))] )

            print(len(scores_of_iD_data), len(scores_of_OOD_data))

            fpr, tpr, threshs = roc_curve(GTs, scores_of_iD_data + scores_of_OOD_data)
            auroc = roc_auc_score(GTs, scores_of_iD_data + scores_of_OOD_data)
            
            descending_scores = np.sort(scores_of_iD_data)[::-1]
            thresh_95_percentile = descending_scores[ int(0.95*len(scores_of_iD_data)) ]
            # all OOD scores less than above thresh will be correctly classified as OOD (So this gives us TNR @ 95% TPR)
            correctly_detected_OOD = 0
            for s in scores_of_OOD_data:
                if s <= thresh_95_percentile:
                    correctly_detected_OOD += 1
            tnr = correctly_detected_OOD/len(scores_of_OOD_data)

            print(f'AUROC {auroc}, TNR @ 95% TPR {tnr}')
            
