import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchattacks import APGD, GN, PGD
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy import stats
from glob import glob
from PIL import Image
from BPDA import BPDA_quantization

def get_timestamp():
    format_str = "%A, %d %b %Y %H:%M:%S %p"
    result = datetime.now().strftime(format_str)
    return result

def tensor2npimg(tensor):
    return tensor.cpu().numpy().transpose((1, 2, 0))

def clean_eval(model, test_loader, device, args):
    model.eval()

    correct = 0
    total = 0
    loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
    with torch.no_grad():
        for images, labels in loader:
            
            images = images.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
        
    return (100 * float(correct) / total)

def pgd_eval(model, test_loader, device, args, save_dir=None):
    model.eval()

    atk = PGD(model, eps=8/255, alpha=0.8/255, steps=20, random_start=True)

    correct = 0
    total = 0
    i = 0
    loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
    for images, labels in loader:
        
        images = atk(images, labels).to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()

        if save_dir is not None:
            for img_idx, img in enumerate(images):
                i += 1
                np_img = tensor2npimg(img)
                four_digit_label = "{0:04}".format(labels[img_idx].item())
                os.makedirs(f"{save_dir}/{four_digit_label}", exist_ok=True)
                plt.imsave(f"{save_dir}/{four_digit_label}/{i}.png", np_img)
        
    return (100 * float(correct) / total)

def pgd_eval_for_epsilon(model, test_loader, device, epsilon, args, save_dir=None):
    model.eval()

    atk = PGD(model, eps=epsilon/255, alpha=0.8/255 * epsilon/8, steps=20, random_start=True)

    correct = 0
    total = 0
    i = 0
    loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
    for images, labels in loader:
        
        images = atk(images, labels).to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()

        if save_dir is not None:
            for img_idx, img in enumerate(images):
                i += 1
                np_img = tensor2npimg(img)
                four_digit_label = "{0:04}".format(labels[img_idx].item())
                os.makedirs(f"{save_dir}/{four_digit_label}", exist_ok=True)
                plt.imsave(f"{save_dir}/{four_digit_label}/{i}.png", np_img)
        
    return (100 * float(correct) / total)

###################################
# Section 4 #
###################################

def get_wts_2x2(local_difference_filter_2x2, channels):
    output = []
    for j in range(channels):
        # creating (channels, 2, 2) matrix
        fil = [ [[0, 0], [0, 0]] ]*channels
        fil[j] = local_difference_filter_2x2
        # creating (channels, channels, 2, 2) matrix
        output.append(fil)
    return torch.tensor(output)

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def penalty_prob(W, device, model_type, metric='mean', total_num_curves = 17): # W = (out_channels = 64, in_channels = 3, filter_size = 3, filter_size = 3)
    out_channels, in_channels, h, w = W.shape[0], W.shape[1], W.shape[2], W.shape[3]

    # diag1_wts = get_wts_2x2([[1.0, 0], [0, -1.0]], in_channels).to(device)
    # diag2_wts = get_wts_2x2([[0, 1.0], [-1.0, 0]], in_channels).to(device)
    # top_wts = get_wts_2x2([[1.0, -1.0], [0, 0]], in_channels).to(device)
    # bottom_wts = get_wts_2x2([[0, 0], [1.0, -1.0]], in_channels).to(device)
    # left_wts = get_wts_2x2([[1.0, 0], [-1.0, 0]], in_channels).to(device)
    # right_wts = get_wts_2x2([[0, 1.0], [0, -1.0]], in_channels).to(device)

    def plot_hist(axis, T, description, i, j):
        oneD_T = torch.abs(T).view(-1).cpu().detach().numpy()
        axis[i, j].hist(oneD_T)
        axis[i, j].set_title(description)
        # axs[i, j].ylabel('number of absolute differences in each bucket')
        # axs[i, j].xlabel('absolute difference buckets')
        return

    # fig, axs = plt.subplots(3, 2, figsize=(7,10))
    # plot_hist(axs, F.conv2d(W, diag1_wts), 'right diagonal', 0, 0)
    # plot_hist(axs, F.conv2d(W, diag2_wts), 'left diag', 0, 1)
    # plot_hist(axs, F.conv2d(W, top_wts), 'top', 1, 0)
    # plot_hist(axs, F.conv2d(W, bottom_wts), 'bottom', 1, 1)
    # plot_hist(axs, F.conv2d(W, left_wts), 'left', 2, 0)
    # plot_hist(axs, F.conv2d(W, right_wts), 'right', 2, 1)
    # os.makedirs(f'./plots', exist_ok=True)
    # plt.savefig(f'./plots/test_filters_hist_{model_type}.png')

    # def get_min_max_differences(Wt):
    #     W2 = torch.zeros(out_channels, in_channels, 1)
    #     for i in range(out_channels):
    #         for j in range(in_channels):
    #             W2[i][j] = torch.max(Wt[i][j]) - torch.min(Wt[i][j])
    #     return W2

    def get_eig_diff(Wt):
        W2 = torch.zeros(out_channels, in_channels, 1)
        for i in range(out_channels):
            for j in range(in_channels):
                # eigenvalues = torch.linalg.eigvals(Wt[i][j]).real # since only real eigenvalues
                # W2[i][j] = torch.max(eigenvalues) - torch.min(eigenvalues)
                W2[i][j] = torch.max(torch.abs(Wt[i][j]))
        return W2

    
    # f, ax = plt.subplots(figsize=(4,4))
    W_new = get_eig_diff(W)
    # oneD_W_new = torch.abs(W_new).view(-1).cpu().detach().numpy()

    # if 'first' in model_type:
    #     ax.set_ylim([0, 185])
    #     bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    # elif 'last' in model_type:
    #     ax.set_ylim([0, 220000])
    #     ax.yaxis.set_major_formatter(human_format)
    #     bins = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]

    # ax.hist(oneD_W_new, bins=bins)
    # eps_value = model_type.split('-')[0] 
    # ax.set_title(f'{eps_value}-robust model')
    # os.makedirs(f'./plots', exist_ok=True)
    # plt.savefig(f'./plots/eigenvalue_diffs_hist_{model_type}.png')

    def second_metric(W_temp):
        if metric == 'mean':
            return torch.mean(W_temp).cpu().item()
        elif metric == 'max':
            return torch.max(W_temp).cpu().item()
            
    return second_metric(W_new)

def penalty_prob_wrn(W, device, model_type, metric='mean', total_num_curves = 25): 
    out_channels, in_channels, h, w = W.shape[0], W.shape[1], W.shape[2], W.shape[3]


    def get_eig_diff(Wt):
        W2 = torch.zeros(out_channels, in_channels, 1)
        for i in range(out_channels):
            for j in range(in_channels):
                # eigenvalues = torch.linalg.eigvals(Wt[i][j]).real # since only real eigenvalues
                # W2[i][j] = torch.max(eigenvalues) - torch.min(eigenvalues)
                W2[i][j] = torch.max(torch.abs(Wt[i][j]))
        return W2
    
    # f, ax = plt.subplots(figsize=(4,4))
    W_new = get_eig_diff(W)
    # oneD_W_new = torch.abs(W_new).view(-1).cpu().detach().numpy()

    # if 'first' in model_type:
    #     ax.set_ylim([0, 26])
    # if 'last' in model_type:
    #     ax.set_ylim([0, 420000])
    #     ax.yaxis.set_major_formatter(human_format)

    # ax.hist(oneD_W_new)
    # eps_value = model_type.split('-')[0] 
    # ax.set_title(f'{eps_value}-robust model')
    # os.makedirs(f'./plots', exist_ok=True)
    # plt.savefig(f'./plots/wrn_eigenvalue_diffs_hist_{model_type}.png')
            
    def second_metric(W_temp):
        if metric == 'mean':
            return torch.mean(W_temp).cpu().item()
        elif metric == 'max':
            return torch.max(W_temp).cpu().item()
            
    return second_metric(W_new)

def penalty_ratio(W, device, model_type): # W = (out_channels = 64, in_channels = 3, filter_size = 3, filter_size = 3)
    out_channels, in_channels, h, w = W.shape[0], W.shape[1], W.shape[2], W.shape[3]

    def get_eig_ratio(Wt):
        W2 = torch.zeros(out_channels, in_channels, 1)
        for i in range(out_channels):
            for j in range(in_channels):
                eigenvalues = torch.linalg.eigvals(Wt[i][j]).real # since only real eigenvalues
                W2[i][j] = torch.min(eigenvalues) / torch.max(eigenvalues)
        return W2

    
    f, ax = plt.subplots(figsize=(4,4))
    W_new = get_eig_ratio(W)
    oneD_W_new = torch.abs(W_new).view(-1).cpu().detach().numpy()

    if 'first' in model_type:
        ax.set_ylim([0, 80])
    if 'last' in model_type:
        ax.set_ylim([0, 140000])
        ax.yaxis.set_major_formatter(human_format)

    bins = [0, 0.25, 0.8, 0.95, 1.05, 1.2, 1.5, 2]
    ax.hist(oneD_W_new, bins=bins)
    eps_value = model_type.split('-')[0] 
    ax.set_title(f'{eps_value}-robust model')
    os.makedirs(f'./plots', exist_ok=True)
    plt.savefig(f'./plots/eigenvalue_ratios_hist_{model_type}.png')
            
    return 1 

def penalty_ratio_wrn(W, device, model_type): 
    out_channels, in_channels, h, w = W.shape[0], W.shape[1], W.shape[2], W.shape[3]

    def get_eig_ratio(Wt):
        W2 = torch.zeros(out_channels, in_channels, 1)
        for i in range(out_channels):
            for j in range(in_channels):
                eigenvalues = torch.linalg.eigvals(Wt[i][j]).real # since only real eigenvalues
                W2[i][j] = torch.min(eigenvalues) / torch.max(eigenvalues)
        return W2
    
    f, ax = plt.subplots(figsize=(4,4))
    W_new = get_eig_ratio(W)
    oneD_W_new = torch.abs(W_new).view(-1).cpu().detach().numpy()

    if 'first' in model_type:
        ax.set_ylim([0, 20])
    if 'last' in model_type:
        ax.set_ylim([0, 150000])
        ax.yaxis.set_major_formatter(human_format)

    bins = [0, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2]
    ax.hist(oneD_W_new, bins=bins)
    eps_value = model_type.split('-')[0] 
    ax.set_title(f'{eps_value}-robust model')
    os.makedirs(f'./plots', exist_ok=True)
    plt.savefig(f'./plots/wrn_eigenvalue_ratios_hist_{model_type}.png')
            
    return 1 

###################################
# Section 5 #
###################################

def scaled_floor(feature, scale):
    # print(torch.max(feature), torch.min(feature))
    return torch.floor(scale*feature)/scale

def quantize(feature, scale):
    return scaled_floor(feature, scale)

def channelwise_quantize_first_K_channels(feat_r, K, scale):
    for idx in range(K):
        feat_r[idx] = quantize(feat_r[idx], scale)
    return feat_r

def special_eval_with_two_part_model(first, second, test_loader, device, args, scale, K):
    correct = 0
    total = 0
    loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
        
            intermediate_feat = first(images)[0] # since batch size for quantization is 1!
            # print(f'Tot_channels: ', intermediate_feat.shape)
            quantized_feat = channelwise_quantize_first_K_channels(intermediate_feat, K, scale).unsqueeze(0)
            outputs = second(quantized_feat)
        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()

    return (100 * float(correct) / total)

def special_eval_with_two_part_model_generic(first, second, test_loader, device, args, scale):
    correct = 0
    total = 0
    loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
        
            intermediate_feat = first(images)

            quantized_feat = quantize(intermediate_feat, scale)
            outputs = second(quantized_feat)
        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()

    return (100 * float(correct) / total)    

def many_feature_quant_eval(model, inp_test_loader, epsilon_test, device, data_dir, model_name, args, scale):
    model.eval()

    save_dir = f"{data_dir}/pgd_{model_name}"
    print(f'checking for {save_dir}..')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        pretrained_pgd20_acc = pgd_eval_for_epsilon(model, inp_test_loader, device, epsilon_test, args, save_dir=save_dir)
        print(f'Robust_Pretrained models PGD20 accuracy is {pretrained_pgd20_acc}.')

    print(f'loading from {save_dir}..')
    test_set = torchvision.datasets.ImageFolder(save_dir, transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    for feat_idx, (first, second, tot_channels) in enumerate([(model.before_f0, model.after_f0, 64), 
                        (model.before_f1, model.after_f1, 64),
                        (model.before_f2, model.after_f2, 128),
                        (model.before_f3, model.after_f3, 256),
                        (model.before_f4_avgpool, model.after_f4_avgpool, 512),
                        ]):
        for K in range(tot_channels):
            acc = special_eval_with_two_part_model(first, second, test_loader, device, args, scale, K)
            print(f'{model_name}, feat_idx {feat_idx}, K {K}, BPDA accuracy {acc}')     
    return 1

def many_feature_full_quant_eval(model, inp_test_loader, epsilon_test, device, data_dir, model_name, args, scale):
    model.eval()

    if epsilon_test == 0:
        pretrained_clean_acc = clean_eval(model, inp_test_loader, device, args)
        print(f'Robust_Pretrained models clean accuracy is {pretrained_clean_acc}.')
        for feat_idx, (first, second, tot_channels) in enumerate([(model.before_f0, model.after_f0, 64), 
                            (model.before_f1, model.after_f1, 64),
                            (model.before_f2, model.after_f2, 128),
                            (model.before_f3, model.after_f3, 256),
                            (model.before_f4_avgpool, model.after_f4_avgpool, 512),
                            ]):
            acc = special_eval_with_two_part_model_generic(first, second, inp_test_loader, device, args, scale)
            print(f'{model_name}, feat_idx {feat_idx}, clean accuracy {acc}')     
        return 1

    else:
        save_dir = f"{data_dir}/pgd_{model_name}"
        print(f'checking for {save_dir}..')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            pretrained_pgd20_acc = pgd_eval_for_epsilon(model, inp_test_loader, device, epsilon_test, args, save_dir=save_dir)
            print(f'Robust_Pretrained models PGD20 accuracy is {pretrained_pgd20_acc}.')

        print(f'loading from {save_dir}..')
        test_set = torchvision.datasets.ImageFolder(save_dir, transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

        for feat_idx, (first, second, tot_channels) in enumerate([(model.before_f0, model.after_f0, 64), 
                            (model.before_f1, model.after_f1, 64),
                            (model.before_f2, model.after_f2, 128),
                            (model.before_f3, model.after_f3, 256),
                            (model.before_f4_avgpool, model.after_f4_avgpool, 512),
                            ]):
            K = tot_channels
            acc = special_eval_with_two_part_model(first, second, test_loader, device, args, scale, K)
            print(f'{model_name}, feat_idx {feat_idx}, K {K}, BPDA accuracy {acc}')     
        return 1

def many_feature_full_quant_eval_BPDA(model, test_loader, epsilon, device, data_dir, model_name, args, scale):
    model.eval()

    for feat_idx, (first, second, tot_channels) in enumerate([(model.before_f0, model.after_f0, 64), 
                        (model.before_f1, model.after_f1, 64),
                        (model.before_f2, model.after_f2, 128),
                        (model.before_f3, model.after_f3, 256),
                        (model.before_f4_avgpool, model.after_f4_avgpool, 512),
                        ]):
        
        atk = BPDA_quantization(model, first, second, scale, eps=epsilon/255, alpha=0.8/255 * epsilon/8, steps=20, random_start=True)

        correct = 0
        total = 0
        loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
        for batch_idx, (images, labels) in enumerate(loader):
            images = atk(images, labels).to(device)
            intermediate_feat = first(images)
            quantized_feat = quantize(intermediate_feat, scale)
            outputs = second(quantized_feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
        
        acc = (100 * float(correct) / total)
        print(f'{model_name}, feat_idx {feat_idx}, BPDA accuracy {acc}')     
    return 1

def many_feature_full_quant_eval_generic(model, inp_test_loader, epsilon_test, device, data_dir, model_name, args, scale):
    model.eval()

    if epsilon_test == 0:
        pretrained_clean_acc = clean_eval(model, inp_test_loader, device, args)
        print(f'Robust_Pretrained models clean accuracy is {pretrained_clean_acc}.')
        for feat_idx, (first, second) in enumerate([
                            (model.before_f0, model.after_f0), 
                            (model.before_f1, model.after_f1),
                            (model.before_f2, model.after_f2),
                            (model.before_f3, model.after_f3),
                            ]):
            acc = special_eval_with_two_part_model_generic(first, second, inp_test_loader, device, args, scale)
            print(f'{model_name}, feat_idx {feat_idx}, clean accuracy {acc}')     
        return 1

    else:
        save_dir = f"{data_dir}/pgd_{model_name}"
        print(f'checking for {save_dir}..')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            pretrained_pgd20_acc = pgd_eval_for_epsilon(model, inp_test_loader, device, epsilon_test, args, save_dir=save_dir)
            print(f'Robust_Pretrained models PGD20 accuracy is {pretrained_pgd20_acc}.')

        print(f'loading from {save_dir}..')
        test_set = torchvision.datasets.ImageFolder(save_dir, transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        for feat_idx, (first, second) in enumerate([
                            (model.before_f0, model.after_f0), 
                            (model.before_f1, model.after_f1),
                            (model.before_f2, model.after_f2),
                            (model.before_f3, model.after_f3),
                            ]):
            acc = special_eval_with_two_part_model_generic(first, second, test_loader, device, args, scale)
            print(f'{model_name}, feat_idx {feat_idx}, BPDA accuracy {acc}')     
        return 1


def many_feature_full_quant_eval_generic_BPDA(model, test_loader, epsilon, device, data_dir, model_name, args, scale):
    model.eval()

    for feat_idx, (first, second) in enumerate([
                        (model.before_f0, model.after_f0), 
                        (model.before_f1, model.after_f1),
                        (model.before_f2, model.after_f2),
                        (model.before_f3, model.after_f3),
                        ]):
        
        atk = BPDA_quantization(model, first, second, scale, eps=epsilon/255, alpha=0.8/255 * epsilon/8, steps=20, random_start=True)

        correct = 0
        total = 0
        loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
        for batch_idx, (images, labels) in enumerate(loader):
            images = atk(images, labels).to(device)
            intermediate_feat = first(images)
            quantized_feat = quantize(intermediate_feat, scale)
            outputs = second(quantized_feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
        
        acc = (100 * float(correct) / total)
        print(f'{model_name}, feat_idx {feat_idx}, BPDA accuracy {acc}')     
    return 1


###################################
# Section 6 #
###################################



###################################
# Section 7 #
###################################

def get_means_and_nums_of_pre_relu_activations_on_corruptions(model, test_set, device, corruption, args, model_type, USE_SAVED_NPS = False): 
    def n_zero(arr):
        return np.sum(arr == 0)

    def n_pos(arr):
        return np.sum(arr > 0)

    def n_neg(arr):
        return np.sum(arr < 0)
    
    model.eval()

    store_0 = []; store_0_pre_bn = []; store_1 = []; store_2 = []; store_3 = []; store_4 = []

    extra_text = ''
    if args.no_bn:
        extra_text = '_no_bn'

    if USE_SAVED_NPS:
        store_0_pre_bn_1D = np.load(f'./nps_activations{extra_text}/pre_bn_and_relu_0_tested_on_{corruption}_{model_type}.npy')
        store_0_1D = np.load(f'./nps_activations{extra_text}/pre_relu_0_tested_on_{corruption}_{model_type}.npy')
        store_1_1D = np.load(f'./nps_activations{extra_text}/pre_relu_1_tested_on_{corruption}_{model_type}.npy')
        store_2_1D = np.load(f'./nps_activations{extra_text}/pre_relu_2_tested_on_{corruption}_{model_type}.npy')
        store_3_1D = np.load(f'./nps_activations{extra_text}/pre_relu_3_tested_on_{corruption}_{model_type}.npy')
        store_4_1D = np.load(f'./nps_activations{extra_text}/pre_relu_4_tested_on_{corruption}_{model_type}.npy')
    else:
        base_path = '../data/CIFAR-10-C/'
        test_set.data = np.load(base_path + corruption + '.npy')[40000:]
        test_set.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))[40000:]

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=args.batch_size,
                                         shuffle=False)
        loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
        for images, labels in loader:            
            images = images.to(device)
            pre_relu_and_bn_0, pre_relu_0, pre_relu_1, pre_relu_2, pre_relu_3, pre_relu_4, _, out = model.forward_with_all_feature(images)
            store_0_pre_bn.append(pre_relu_and_bn_0.view(-1).cpu().detach())
            store_0.append(pre_relu_0.view(-1).cpu().detach())
            store_1.append(pre_relu_1.view(-1).cpu().detach())
            store_2.append(pre_relu_2.view(-1).cpu().detach())
            store_3.append(pre_relu_3.view(-1).cpu().detach())
            store_4.append(pre_relu_4.view(-1).cpu().detach())            

        store_0_pre_bn_1D = torch.cat(store_0_pre_bn).numpy()
        store_0_1D = torch.cat(store_0).numpy()
        store_1_1D = torch.cat(store_1).numpy()
        store_2_1D = torch.cat(store_2).numpy()
        store_3_1D = torch.cat(store_3).numpy()
        store_4_1D = torch.cat(store_4).numpy()
        os.makedirs(f'./nps_activations{extra_text}', exist_ok=True)
        np.save(f'./nps_activations{extra_text}/pre_bn_and_relu_0_tested_on_{corruption}_{model_type}', store_0_pre_bn_1D)
        np.save(f'./nps_activations{extra_text}/pre_relu_0_tested_on_{corruption}_{model_type}', store_0_1D)
        np.save(f'./nps_activations{extra_text}/pre_relu_1_tested_on_{corruption}_{model_type}', store_1_1D)
        np.save(f'./nps_activations{extra_text}/pre_relu_2_tested_on_{corruption}_{model_type}', store_2_1D)
        np.save(f'./nps_activations{extra_text}/pre_relu_3_tested_on_{corruption}_{model_type}', store_3_1D)
        np.save(f'./nps_activations{extra_text}/pre_relu_4_tested_on_{corruption}_{model_type}', store_4_1D)

    return (np.mean(store_0_pre_bn_1D), stats.mode(store_0_pre_bn_1D), np.median(store_0_pre_bn_1D), np.std(store_0_pre_bn_1D),
            np.mean(store_0_1D), stats.mode(store_0_1D), np.median(store_0_1D), np.std(store_0_1D), 
            np.mean(store_1_1D), stats.mode(store_1_1D), np.median(store_1_1D), np.std(store_1_1D), 
            np.mean(store_2_1D), stats.mode(store_2_1D), np.median(store_2_1D), np.std(store_2_1D), 
            np.mean(store_3_1D), stats.mode(store_3_1D), np.median(store_3_1D), np.std(store_3_1D), 
            np.mean(store_4_1D), stats.mode(store_4_1D), np.median(store_4_1D), np.std(store_4_1D),
            n_zero(store_0_pre_bn_1D), n_pos(store_0_pre_bn_1D), n_neg(store_0_pre_bn_1D),
            n_zero(store_0_1D), n_pos(store_0_1D), n_neg(store_0_1D),
            n_zero(store_1_1D), n_pos(store_1_1D), n_neg(store_1_1D),
            n_zero(store_2_1D), n_pos(store_2_1D), n_neg(store_2_1D),
            n_zero(store_3_1D), n_pos(store_3_1D), n_neg(store_3_1D),
            n_zero(store_4_1D), n_pos(store_4_1D), n_neg(store_4_1D))

def get_means_and_nums_of_pre_relu_activations_on_corruptions_wrn(model, test_set, device, corruption, args, model_type, USE_SAVED_NPS = False): 
    def n_zero(arr):
        return np.sum(arr == 0)

    def n_pos(arr):
        return np.sum(arr > 0)

    def n_neg(arr):
        return np.sum(arr < 0)
    
    model.eval()

    store_0 = []; store_1 = []; store_2 = []; store_3 = []; store_4 = []

    extra_text = ''
    if args.no_bn:
        extra_text = '_no_bn'

    if USE_SAVED_NPS:
        store_0_1D = np.load(f'./nps_activations{extra_text}/wrn_pre_relu_0_tested_on_{corruption}_{model_type}.npy')
        store_1_1D = np.load(f'./nps_activations{extra_text}/wrn_pre_relu_1_tested_on_{corruption}_{model_type}.npy')
        store_2_1D = np.load(f'./nps_activations{extra_text}/wrn_pre_relu_2_tested_on_{corruption}_{model_type}.npy')
        store_3_1D = np.load(f'./nps_activations{extra_text}/wrn_pre_relu_3_tested_on_{corruption}_{model_type}.npy')
        store_4_1D = np.load(f'./nps_activations{extra_text}/wrn_pre_relu_4_tested_on_{corruption}_{model_type}.npy')
    else:
        base_path = '../data/CIFAR-10-C/'
        test_set.data = np.load(base_path + corruption + '.npy')[40000:]
        test_set.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))[40000:]

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=args.batch_size,
                                         shuffle=False)
        loader = tqdm(test_loader, dynamic_ncols=True) if not args.tqdm_off else test_loader
        for images, labels in loader:            
            images = images.to(device)
            out0, out1_inner_pre_relu, out2_inner_pre_relu, out3_inner_pre_relu, out_last_pre_relu, out = model.forward_with_all_feature(images)
            store_0.append(out0.view(-1).cpu().detach())
            store_1.append(out1_inner_pre_relu.view(-1).cpu().detach())
            store_2.append(out2_inner_pre_relu.view(-1).cpu().detach())
            store_3.append(out3_inner_pre_relu.view(-1).cpu().detach())
            store_4.append(out_last_pre_relu.view(-1).cpu().detach())            

        store_0_1D = torch.cat(store_0).numpy()
        store_1_1D = torch.cat(store_1).numpy()
        store_2_1D = torch.cat(store_2).numpy()
        store_3_1D = torch.cat(store_3).numpy()
        store_4_1D = torch.cat(store_4).numpy()
        os.makedirs(f'./nps_activations{extra_text}', exist_ok=True)
        np.save(f'./nps_activations{extra_text}/wrn_pre_relu_0_tested_on_{corruption}_{model_type}', store_0_1D)
        np.save(f'./nps_activations{extra_text}/wrn_pre_relu_1_tested_on_{corruption}_{model_type}', store_1_1D)
        np.save(f'./nps_activations{extra_text}/wrn_pre_relu_2_tested_on_{corruption}_{model_type}', store_2_1D)
        np.save(f'./nps_activations{extra_text}/wrn_pre_relu_3_tested_on_{corruption}_{model_type}', store_3_1D)
        np.save(f'./nps_activations{extra_text}/wrn_pre_relu_4_tested_on_{corruption}_{model_type}', store_4_1D)

    return (np.mean(store_0_1D), stats.mode(store_0_1D), np.median(store_0_1D), np.std(store_0_1D), 
            np.mean(store_1_1D), stats.mode(store_1_1D), np.median(store_1_1D), np.std(store_1_1D), 
            np.mean(store_2_1D), stats.mode(store_2_1D), np.median(store_2_1D), np.std(store_2_1D), 
            np.mean(store_3_1D), stats.mode(store_3_1D), np.median(store_3_1D), np.std(store_3_1D), 
            np.mean(store_4_1D), stats.mode(store_4_1D), np.median(store_4_1D), np.std(store_4_1D),
            n_zero(store_0_1D), n_pos(store_0_1D), n_neg(store_0_1D),
            n_zero(store_1_1D), n_pos(store_1_1D), n_neg(store_1_1D),
            n_zero(store_2_1D), n_pos(store_2_1D), n_neg(store_2_1D),
            n_zero(store_3_1D), n_pos(store_3_1D), n_neg(store_3_1D),
            n_zero(store_4_1D), n_pos(store_4_1D), n_neg(store_4_1D))


###################################
# Section 8 #
###################################

def get_baseline_scores(model, clean_test_loader, epsilon, epsilon_test, Temp, device, args):
    scores_on_iD_data = []
    for e in range(int(epsilon)+1):
        scores_on_iD_data.extend( baseline_scores(model, clean_test_loader, float(e), Temp, device, args) )

    scores_on_OOD_data = []
    scores_on_OOD_data.extend( baseline_scores(model, clean_test_loader, epsilon_test, Temp, device, args) )

    return scores_on_iD_data, scores_on_OOD_data


def baseline_scores(model, clean_test_loader, epsilon, Temp, device, args):
    model.eval()

    if epsilon > 0:
        atk = PGD(model, eps=epsilon/255, alpha=0.8/255 * epsilon/8, steps=20, random_start=True)

    softmax = nn.Softmax()
    correct = 0
    total = 0
    scores = []
    loader = tqdm(clean_test_loader, dynamic_ncols=True) if not args.tqdm_off else clean_test_loader
    for images, labels in loader:
        
        if epsilon == 0:
            images = images.to(device)
        elif epsilon > 0:
            images = atk(images, labels).to(device)

        outputs = model(images)
        probs = softmax(outputs.detach().clone()/Temp)
        
        max_probs, predicted = torch.max(probs.data, 1)
        scores.extend(max_probs.cpu().tolist())
        
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
        
    return scores

def mahala_dist(F, MU, PRECISION):
    return (F - MU).double() @ PRECISION.double() @ (F - MU).T.double()

def get_Mahala_scores(model, clean_test_loader, epsilon, epsilon_test, device, args):
    scores_on_iD_data = []
    scores_on_OOD_data = []

    return scores_on_iD_data, scores_on_OOD_data