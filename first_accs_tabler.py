import numpy as np
import matplotlib.pyplot as plt 
import os 
# from adjustText import adjust_text # see: https://github.com/Phlya/adjustText and post at https://stackoverflow.com/a/34762716

for model_name in ['wrn', 'resnet18']: # 'wrn' or 'resnet18'

    possible_types = ['']
    # if model_name == 'resnet18':
    #     possible_types += ['Stacked']
    
    for type in possible_types: 
        if model_name == 'wrn':
            deltaz = [(1, 3), (2, 4), (5, 7), (6, 8)]
            all_deltaz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
            epsilonz = list(range(0, 14, 2))
        else:
            if type == 'Stacked':
                deltaz = [(1, 2)] # list(range(0, 14, 2)) # deltas_I_care_about
                all_deltaz = [0, 2, 4, 6, 8, 10, 12]
                epsilonz = list(range(0, 14, 2)) # epsilons_I_care_about
            else:
                deltaz = [(1, 3), (2, 4), (5, 7), (6, 8)] # list(range(0, 13, 1)) # IF YOU CHANGE THIS CHANGE LINES 59-62 where we annotate ALSO
                all_deltaz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                epsilonz = list(range(0, 13, 1))

        accs = {}

        for delta in all_deltaz:
            if model_name == 'wrn':
                filename = f'./logs/wrn_which_of_them_generalizes_best_{delta}by255.log'
            else:
                if type == 'Stacked':
                    filename = f'./logs/which_of_them_generalizes_best_{type}_{delta}by255.log'
                else:
                    filename = f'./logs/which_of_them_generalizes_best_{delta}by255.log'
            accs[delta] = {'train': {}, 'test': {}}
            with open(filename) as file:
                lines = file.readlines()
                for line in lines:
                    if ': TRAIN' in line:
                        acc = float(line.split(' ')[-1])
                        epsilon = int(float(line.split('/')[0].split(' ')[-1]))
                        if epsilon in epsilonz:
                            accs[delta]['train'][epsilon] = acc
                    elif ': TEST' in line:
                        acc = float(line.split(' ')[-1])
                        epsilon = int(float(line.split('/')[0].split(' ')[-1]))
                        if epsilon in epsilonz:
                            accs[delta]['test'][epsilon] = acc

        row_str = 'clean & '
        for epsilon in epsilonz:
            clean_acc = accs[0]['test'][epsilon]
            row_str += f'{clean_acc} & '
        row_str += ' \\\\'
        print(row_str)

        row_str = 'PGD$_{\epsilon}$-20 & '
        for epsilon in epsilonz:
            acc = accs[epsilon]['test'][epsilon]
            row_str += f'{acc} & '
        row_str += ' \\\\ \\hline'
        print(row_str)
        print('\n\n\n')