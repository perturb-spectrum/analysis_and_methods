"""
AT and Feature Quantization
"""

import numpy as np
import torch
import torchvision.utils
import torchattacks
from utils import *
import argparse
from models.dm_wide_resnet import DMWideResNet, Swish, CIFAR10_MEAN, CIFAR10_STD
from models.resnet_ours import ResNet18_preReLU
import pickle
from utils import *
import os

# HYPERPARAMS
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', default='resnet_18', type=str, help='dm_wrn_28_10 or resnet_18')
parser.add_argument('--metric', default='max', type=str, help='max or mean')
parser.add_argument('--model-type', default='', type=str, help=' <empty> or Stacked')
parser.add_argument('--tqdm-off', default=False, action='store_true', help='want to turn off tqdm?')
parser.add_argument('--plot', default=False, action='store_true', help='want to plot')
args = parser.parse_args()


# SEED, VER, GPU
torch.manual_seed(0)
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
device = torch.device("cpu") 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not args.plot:
    store = {'conv0': {}}
    # DEFINE PRETRAINED MODEL and LOAD
    if args.model == 'resnet_18':
        pretrained_nr_model = ResNet18_preReLU()
        pretrained_nr_model.load_state_dict(torch.load('./saved_state_dicts/ST_resnet18/ep_19.pth'), strict=True)
        pretrained_nr_model = pretrained_nr_model.to(device)
        os.makedirs(f'./plots', exist_ok=True)

        count = 0
        epsilonz = [0,0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] if args.model_type == 'Stacked' else [0,0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        total_num_curves = 17
        for eps in epsilonz:
            count += 1

            if eps == 0: 
                store['conv0'][eps] = penalty_prob(pretrained_nr_model.conv1.weight, device, f'non-{args.model_type}-robust-first', args.metric)
                i = 1
                for l_idx, l in enumerate([pretrained_nr_model.layer1, pretrained_nr_model.layer2, pretrained_nr_model.layer3, pretrained_nr_model.layer4]):
                    for idx in [0, 1]:
                        for conv_idx, conv in enumerate([l[idx].conv1, l[idx].conv2]):
                            key = f'layer{l_idx+1}[{idx}].conv{conv_idx+1}'
                            if not key in store:
                                store[key] = {}
                            i += 1
                            store[key][eps] = penalty_prob(conv.weight, device, f'', args.metric)
                # print(f'total {i}') # 17
                # exit()
            else:
                pretrained_r_model = ResNet18_preReLU() 
                pretrained_r_model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_resnet18_{eps}/ep_76.pth'), strict=True)
                pretrained_r_model = pretrained_r_model.to(device)
                store['conv0'][eps] = penalty_prob(pretrained_r_model.conv1.weight, device, f'{eps}-{args.model_type}-robust-first', args.metric)
                for l_idx, l in enumerate([pretrained_r_model.layer1, pretrained_r_model.layer2, pretrained_r_model.layer3, pretrained_r_model.layer4]):
                    for idx in [0, 1]:
                        for conv_idx, conv in enumerate([l[idx].conv1, l[idx].conv2]):
                            key = f'layer{l_idx+1}[{idx}].conv{conv_idx+1}'
                            if not key in store:
                                store[key] = {}
                            store[key][eps] = penalty_prob(conv.weight, device, f'', args.metric)


    elif args.model == 'wrn':
        pretrained_nr_model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        
        pretrained_nr_model.load_state_dict(torch.load('./saved_state_dicts/ST_wrn/ep_18.pth'), strict=True)
        pretrained_nr_model = pretrained_nr_model.to(device)
        os.makedirs(f'./plots', exist_ok=True)

        count = 0
        total_num_curves = 25
        for eps in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
            count += 1

            if eps == 0: 
                store['conv0'][eps] = penalty_prob_wrn(pretrained_nr_model.init_conv.weight, device, f'non-{args.model_type}-robust-first', args.metric)
                l = pretrained_nr_model.layer
                i = 1
                for l_idx in [0, 1, 2]:
                    for block_idx in [0, 1, 2, 3]:
                        for conv_idx, conv in enumerate([l[l_idx].block[block_idx].conv_0, l[l_idx].block[block_idx].conv_1]):
                            key = f'layer[{l_idx}].block[{block_idx}].conv_{conv_idx}'
                            if not key in store:
                                store[key] = {}
                            store[key][eps] = penalty_prob_wrn(conv.weight, device, f'', args.metric)
                            i += 1
                # print(f'total {i}') # 25
                # exit()
            else: 
                pretrained_r_model = DMWideResNet(num_classes=10, depth=28, width=10, activation_fn=Swish, mean=CIFAR10_MEAN, std=CIFAR10_STD)
                pretrained_r_model.load_state_dict(torch.load(f'./saved_state_dicts/{args.model_type}PGDAT_wrn_{eps}/ep_80.pth'), strict=True)
                pretrained_r_model = pretrained_r_model.to(device)
                store['conv0'][eps] = penalty_prob_wrn(pretrained_r_model.init_conv.weight, device, f'{eps}-{args.model_type}-robust-first', args.metric)
                l = pretrained_r_model.layer
                for l_idx in [0, 1, 2]:
                    for block_idx in [0, 1, 2, 3]:
                        for conv_idx, conv in enumerate([l[l_idx].block[block_idx].conv_0, l[l_idx].block[block_idx].conv_1]):
                            key = f'layer[{l_idx}].block[{block_idx}].conv_{conv_idx}'
                            if not key in store:
                                store[key] = {}
                            store[key][eps] = penalty_prob_wrn(conv.weight, device, f'', args.metric)

    os.makedirs('./pkls', exist_ok=True)
    with open(f'./pkls/{args.metric}_of_maxs_{args.model}_{args.model_type}.pkl', 'wb') as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif args.plot:
    with open(f'./pkls/{args.metric}_of_maxs_{args.model}_{args.model_type}.pkl', 'rb') as handle:
        store = pickle.load(handle)

    if args.model == 'resnet_18':
        total_num_curves = 17
        plt.figure(figsize=(8,7))
    else:
        total_num_curves = 25
        plt.figure(figsize=(10,7))
    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, total_num_curves)))
    for idx, (layer_name, trend) in enumerate(store.items()):
        if idx == 0:
            linestyle = '-o'
        elif idx == 1:
            linestyle = '-^'
        else:
            linestyle = '--'
        c = next(color)
        plt.plot(list(trend.keys()), list(trend.values()), linestyle, label=layer_name, c=c)
    plt.legend(loc="right")
    nice_name = {'resnet_18': 'ResNet18', 'wrn': 'WideResNet-28-10'}
    plt.title(f'{nice_name[args.model]}', fontsize=24)
    plt.ylabel(args.metric+'$_{w \in \; conv \;\; i}}$ ||w||$_{\infty}$', fontsize=24)
    plt.xlabel(f'$\epsilon$', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if args.metric == 'max':
        plt.savefig(f'./plots/all_{args.metric}_of_maxs_{args.model}_{args.model_type}.png')
    else:
        plt.savefig(f'./plots/all_{args.metric}_of_maxs_{args.model}_{args.model_type}.png', bbox_inches='tight')
    
