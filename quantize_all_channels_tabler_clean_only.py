import numpy as np
import os

def print_table(table):
    for row in table:
        print(row)
    print('\n\n')

for model_name in ['resnet18', 'wrn']: # 'resnet18' or 'wrn'
    print(model_name, '\n\n\n\n')
    for SCALE in [4, 6, 8, 10, 12]:
        # print(SCALE, f'\n\n')
        print(f'\n')

        if model_name == 'wrn':
            epsilonz = list(range(0, 14, 2))
            n_feats = 5
        else:
            epsilonz = list(range(0, 13, 1)) 
            n_feats = 6
        deltaz = [0.0] # [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

        # Get accs
        def get_table_for_a_scale(scale):
            table = np.zeros((len(deltaz)*n_feats, len(epsilonz))) 

            for delta in deltaz:
                if model_name == 'wrn':
                    filename = f'./logs/wrn_full_scale_{scale}_eps_test_{int(delta)}.log'
                    feat_idx_to_name = {0: 'none', 1: 'init conv', 2: 'layer[0]', 3: 'layer[1]', 4: 'layer[2]'}
                else:
                    filename = f'./logs/full_scale_{scale}_eps_test_{int(delta)}.log'
                    feat_idx_to_name = {0: 'none', 1: 'conv0', 2: 'layer1', 3: 'layer2', 4: 'layer3', 5: 'layer4'}
                with open(filename) as file:
                    lines = file.readlines()
                    # print(lines)
                    epsilon = -1
                    for line in lines:
                        if 'Robust_Pretrained models' in line:
                            acc = float(line.split(' ')[-1][:-2]) # the [:-2] is to convert '16.28.\n' -> '16.28'
                            table[0, epsilon+1] = acc
                            epsilon += 1
                            feat_idx = 0
                        if 'feat_idx' in line:
                            acc = float(line.split(' ')[-1][:-2]) # the [:-2] is to convert '16.28.\n' -> '16.28'
                            feat_idx += 1
                            table[feat_idx, epsilon] = acc

            return table, feat_idx_to_name

        table_scale_8, feat_idx_to_name_scale_8 = get_table_for_a_scale(8)
        if SCALE == 8:
            table = table_scale_8
            feat_idx_to_name = feat_idx_to_name_scale_8
        else:
            table, feat_idx_to_name = get_table_for_a_scale(SCALE)
            # replace all 'none' accs with the values in table_scale_8
            for delta in deltaz:
                for epsilon in range(-1, len(epsilonz)-1):
                    table[int((delta/2.0-1) * n_feats), epsilon+1] = table_scale_8[int((delta/2.0-1) * n_feats), epsilon+1]
                        

        # print
        # print_table(table)

        # Subtract none row from 5 rows after none row and ...
        # Finally latex print
        os.makedirs(f'./nps_tables', exist_ok=True)
        np.save(f'./nps_tables/{model_name}_{SCALE}.npy', table)
        new_table = []
        for delta in deltaz:
            none_row = 0
            for feat_idx in range(n_feats):
                row = none_row + feat_idx
                if row!= none_row:
                    new_row_string = f' & {feat_idx_to_name[feat_idx]} '
                else:
                    new_row_string = f'{int(SCALE)} & {feat_idx_to_name[feat_idx]} '
                for col in range(len(epsilonz)):
                    none_row_val = table[none_row, col]
                    maximum_of_big_row = np.max(table[none_row:none_row+n_feats, col])
                    if row!= none_row:
                        row_val = table[row, col] - table[none_row, col]
                        actual_val = table[row,col]
                        if row_val > 0 and maximum_of_big_row-none_row_val != 0:
                            fraction_of_max = max(7, int((actual_val-none_row_val)/(maximum_of_big_row-none_row_val) * 40))
                            new_row_string += f'& {actual_val:.2f}'  
                        elif row_val < 0 and maximum_of_big_row-none_row_val != 0:
                            fraction_of_max = int(abs((actual_val-none_row_val)/(maximum_of_big_row-none_row_val) * 1))
                            new_row_string += f'& {actual_val:.2f}'
                        else:
                            new_row_string += f'& {actual_val:.2f} '
                    else:
                        row_val = none_row_val
                        new_row_string += f'& {row_val:.2f} '

                new_table.append(new_row_string+' \\\\')
            new_table.append('\\hline')
                
        print_table(new_table)







