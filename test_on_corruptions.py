import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from tqdm import tqdm

from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet50, ResNet18_preReLU
import pickle 
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGDAT')
parser.add_argument('--model-name', default='resnet18', type=str, help=' resnet18 or wrn ')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
args = parser.parse_args()

def test(net, test_loader, device):
  """
  Evaluate network on given dataset (includes segmenting every batch).
  Ref: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/cifar.py#L266
  """
  net.eval()
  total_loss = 0.
  total_correct = 0
  loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
  with torch.no_grad():
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = net(images)
        loss = F.cross_entropy(logits, targets)
        pred = logits.data.max(1)[1]
        total_loss += float(loss.data)
        total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)

def test_c(net, test_data, base_path, CORRUPTIONS, batch_size, device, descriptor):
  """
  Evaluate network on given corrupted dataset  (calls above funtion 'seg_test' which includes segmenting every batch).
  Ref: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/cifar.py#L266
  """
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    # print(f'Number of images {len(test_data.data)}')
    # print(f'Number of labels {len(test_data.targets)}')

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    test_loss, test_acc = test(net, test_loader, device)
    corruption_accs.append(test_acc)
    print(f'{descriptor}, {corruption}, Test Loss {test_loss:.3f}, Test Error {100 - 100. * test_acc:.3f}')

  return np.mean(corruption_accs)

if __name__ == "__main__":
    # SEED, VER, GPU, CORRUPTIONS
    torch.manual_seed(0)
    print("PyTorch", torch.__version__)
    print("Torchvision", torchvision.__version__)
    print("Torchattacks", torchattacks.__version__)
    print("Numpy", np.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression', 
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    print(f"Corruptions {CORRUPTIONS}")

    # TEST DATA
    batch_size = 256
    cifar10c_base_path = '../data/CIFAR-10-C/'
    cifar_test = dsets.CIFAR10(root='../data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

    if args.model_name == 'wrn':
        epsilonz = list(range(0, 14, 2))
    else:
        if args.model_type == 'Stacked':
            epsilonz = list(range(0, 14, 2)) # epsilons_I_care_about
        else:
            epsilonz = list(range(0, 13, 1))

    for epsilon in epsilonz:
        print(f'\n')
        epsilon = float(epsilon)
        if args.model_name == 'resnet18':
            model = ResNet18_preReLU()
            if epsilon == 0:
                model.load_state_dict(torch.load(f'./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
                descriptor = 'non-robust model'
            else:
                model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{epsilon}/ep_76.pth'), strict=True)
                descriptor = f'{args.model_type}Trained {epsilon}-robust model'
        elif args.model_name == 'wrn':
            model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
            if epsilon == 0:
                model.load_state_dict(torch.load(f'./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
            else:
                model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{epsilon}/ep_80.pth'), strict=True)

        model = model.to(device)
        
        # TEST
        test_c(model, cifar_test, cifar10c_base_path, CORRUPTIONS, batch_size, device, descriptor)