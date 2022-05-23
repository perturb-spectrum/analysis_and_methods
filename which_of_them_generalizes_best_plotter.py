import numpy as np
import matplotlib.pyplot as plt 
import os 
# from adjustText import adjust_text # see: https://github.com/Phlya/adjustText and post at https://stackoverflow.com/a/34762716

for model_name in ['wrn', 'resnet18']: # 'wrn' or 'resnet18'

    for type in ['']: # 'Stacked', 
        if model_name == 'wrn':
            deltaz = [(1, 3), (2, 4), (5, 7), (6, 8)]
            all_deltaz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
            epsilonz = list(range(0, 14, 2))
        else:
            if type == 'Stacked':
                deltaz = [(1, 2)] # list(range(0, 14, 2)) # deltas_I_care_about
                all_deltaz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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

        for tup in deltaz:
            ############### G.Err. Plots

            def G_Err(delta):
                g_err = {}
                for epsilon, test_acc in accs[delta]['test'].items():
                    g_err[epsilon] = round(100 - test_acc, 2) # accs[epsilon]['train'][epsilon] - 
                return g_err

            # Plot
            plt.figure(figsize=(8,6))
            for delta in tup:
                if delta == 0:
                    plt.plot(epsilonz, list(G_Err(0).values()), label = f'error on clean images', marker='o') # delta = 0
                else:
                    plt.plot(epsilonz, list(G_Err(delta).values()), label = f'error on PGD$_{delta}$-20', marker='o')
            # Annotate
            if type == 'Stacked':
                for delta in tup:
                    for epsilon, g_err_val in G_Err(delta).items():
                        plt.annotate(f'{g_err_val}', xy=(epsilon, g_err_val), xytext=(epsilon-0.5, g_err_val+2))#, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
            else:
                sign_0 = 1
                sign_1 = lambda eps: 1
                sign_2 = lambda eps: 1
                if 1 in tup:
                    sign_0 = 1/1.15
                    sign_1 = lambda eps: 1/1.15
                    sign_2 = lambda eps: 1/1.15
                if 5 in tup or 6 in tup:
                    sign_0 = 1/1.15
                    sign_1 = lambda eps: 1/1.3
                    sign_2 = lambda eps: 1
                if model_name == 'wrn':
                    sign_0 = 1/1.5
                    sign_1 = lambda eps: 1/1.5
                    sign_2 = lambda eps: 1/1.5

                # if 5 in tup or 6 in tup:
                    # sign_0 = -1
                    # sign_1 = lambda eps: -1 if eps >= 8 else 1
                
                
                for epsilon, g_err_val in G_Err(tup[0]).items():
                    if epsilon == 0:
                        xy_text = (epsilon-0.5, g_err_val-sign_0*18)
                    elif epsilon == 3 and (5 in tup or 6 in tup or 2 in tup):
                        xy_text = (epsilon-0.5, g_err_val-sign_0*17)
                    elif epsilon % 2 == 0:
                        xy_text = (epsilon-0.5, g_err_val-sign_2(epsilon)*14)
                    else:
                        xy_text = (epsilon-0.5, g_err_val-sign_1(epsilon)*8)
                    plt.annotate(f'{g_err_val}', xy=(epsilon, g_err_val), xytext=xy_text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), fontsize=14)

                for epsilon, g_err_val in G_Err(tup[1]).items():
                    if epsilon == 0:
                        xy_text = (epsilon-0.5, g_err_val+sign_0*6)
                    elif epsilon == 3 and (5 in tup or 6 in tup or 2 in tup):
                        xy_text = (epsilon-0.5, g_err_val+sign_0*17)
                    elif epsilon % 2 == 0:
                        xy_text = (epsilon-0.5, g_err_val+sign_2(epsilon)*14)
                    else:
                        xy_text = (epsilon-0.5, g_err_val+sign_1(epsilon)*8)
                    plt.annotate(f'{g_err_val}', xy=(epsilon, g_err_val), xytext=xy_text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), fontsize=14)

            plt.xlabel('$\epsilon$', fontsize=20)
            plt.ylabel('Robust Error %', fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            nice_name = {'resnet18': 'ResNet18', 'wrn': 'WideResNet-28-10'}
            plt.title(f'{nice_name[model_name]}', fontsize=20)
            plt.ylim([0,100])
            plt.legend(fontsize=18)
            os.makedirs(f'./plots', exist_ok=True)
            
            if model_name == 'wrn':
                plt.savefig(f'./plots/wrn{type}_which_of_them_classifies_best_gen_err_{tup[0]}{tup[1]}.png')
            else:
                plt.savefig(f'./plots/{type}_which_of_them_classifies_best_gen_err_{tup[0]}{tup[1]}.png')
            
            # ############### Acc. Plot

            # # Plot
            # plt.figure()
            # for delta in tup:
            #     if delta == 0:
            #         plt.plot(epsilonz, list(accs[delta]['test'].values()), label = f'Acc. on clean images') # delta = 0
            #     else:
            #         plt.plot(epsilonz, list(accs[delta]['test'].values()), label = f'Acc. on PGD_{delta}-20')
            # # Annotate
            # for delta in tup:
            #     for epsilon, acc_val in accs[delta]['test'].items():
            #         plt.annotate(f'{round(acc_val,2)}', xy=(epsilon, g_err_val))

            # plt.xlabel('$\epsilon$ (of $\epsilon$-robust model)')
            # plt.ylabel('Test Accuracy % (Acc.)')
            # plt.ylim([0,100])
            # plt.legend()
            # os.makedirs(f'./plots', exist_ok=True)
            # # plt.savefig(f'./plots/{type}_which_of_them_classifies_best_test_acc.png')