import numpy as np
import os 

def print_table(table):
    for row in table:
        print(row)
    print('\n\n')

tables = {'Transf': {}, 'BPDA': {}}
for atk in ['Transf', 'BPDA']:
    extra_text = ''
    if atk == 'BPDA':
        extra_text = '_BPDA'
    for model_name in ['resnet18', 'wrn']: # 'resnet18' or 'wrn'
        SCALE= 8 
        file = f'./nps_tables/{model_name}_{SCALE}{extra_text}.npy'
        tables[atk][model_name] = np.load(file)

# print(tables)

n_feats_resnet18 = 6
n_feats_wrn = 5
deltaz = [2, 4, 6, 8, 10, 12]
epsilonz = [2, 4, 8, 12]  

feat_idx_resnet18_to_name = {0: 'none', 1: 'conv0', 2: 'layer1', 3: 'layer2', 4: 'layer3', 5: 'layer4'}
feat_idx_wrn_to_name = {0: 'none', 1: 'init conv', 2: 'layer[0]', 3: 'layer[1]', 4: 'layer[2]', 5: ''}

# Subtract none row from 5 rows after none row and ...
# Finally latex print
new_table = []
for delta in deltaz:
    none_row = int(delta/2.0-1) * n_feats_resnet18
    none_row_2 = int(delta/2.0-1) * n_feats_wrn
    
    for feat_idx in range(n_feats_resnet18):
        row = none_row + feat_idx
        row_2 = none_row_2 + feat_idx

        if row!= none_row:
            new_row_string = f''
        else:
            new_row_string = f'{int(delta)}'

        for col in range(len(epsilonz)*4 + 2):
            # print(row, col)
            if col == 0:
                new_row_string += f' & {feat_idx_resnet18_to_name[feat_idx]}'
            elif col >= 1 and col <= 4:
                none_row_val = tables["Transf"]["resnet18"][none_row, epsilonz[col-1]]
                maximum_of_big_row = np.max(tables["Transf"]["resnet18"][none_row: none_row+n_feats_resnet18, epsilonz[col-1]])
                row_val = tables["Transf"]["resnet18"][row, epsilonz[col-1]]
                
                max_diff = maximum_of_big_row - none_row_val
                diff = row_val - none_row_val
                extra = f''
                if diff > 0:
                    fraction = max(7, int(diff/max_diff * 40)) 
                    extra = f'\\cellcolor{{blue!{fraction}}}'
                elif diff < 0:
                    extra = f'\\cellcolor{{red!6}}'
                if row == none_row:
                    extra = f'\\cellcolor{{gray!10}}'

                new_row_string += f' & {extra}{row_val}'
            elif col >= 5 and col <= 8:
                none_row_val = tables["BPDA"]["resnet18"][none_row, epsilonz[col-5]]
                maximum_of_big_row = np.max(tables["BPDA"]["resnet18"][none_row: none_row+n_feats_resnet18, epsilonz[col-5]])
                row_val = tables["BPDA"]["resnet18"][row, epsilonz[col-5]]
                
                max_diff = maximum_of_big_row - none_row_val
                diff = row_val - none_row_val
                extra = f''
                if diff > 0:
                    fraction = max(7, int(diff/max_diff * 40)) 
                    extra = f'\\cellcolor{{blue!{fraction}}}'
                elif diff < 0:
                    extra = f'\\cellcolor{{red!6}}'
                if row == none_row:
                    extra = f'\\cellcolor{{gray!10}}'

                new_row_string += f' & {extra}{row_val}'
            elif col == 9:
                new_row_string += f' & {feat_idx_wrn_to_name[feat_idx]}'
            elif col >= 10 and col <= 13 and feat_idx != 5:
                none_row_val = tables["Transf"]["wrn"][none_row_2, int(epsilonz[col-10]/2)]
                maximum_of_big_row = np.max(tables["Transf"]["wrn"][none_row_2:none_row_2+n_feats_wrn, int(epsilonz[col-10]/2)])
                row_val = tables["Transf"]["wrn"][row_2, int(epsilonz[col-10]/2)]
                # print(row, col, epsilonz[col-10]/2, row_val)
                
                max_diff = maximum_of_big_row - none_row_val
                diff = row_val - none_row_val
                extra = f''
                if diff > 0:
                    fraction = max(7, int(diff/max_diff * 40)) 
                    extra = f'\\cellcolor{{blue!{fraction}}}'
                elif diff < 0:
                    extra = f'\\cellcolor{{red!6}}'
                if row == none_row:
                    extra = f'\\cellcolor{{gray!10}}'

                new_row_string += f' & {extra}{row_val}'

            elif col >= 14 and col <= 17 and feat_idx != 5:
                none_row_val = tables["BPDA"]["wrn"][none_row_2, int(epsilonz[col-14]/2)]
                maximum_of_big_row = np.max(tables["BPDA"]["wrn"][none_row_2:none_row_2+n_feats_wrn, int(epsilonz[col-14]/2)])
                row_val = tables["BPDA"]["wrn"][row_2, int(epsilonz[col-14]/2)]
                # print(row_2, col, epsilonz[col-14]/2, row_val)
                
                max_diff = maximum_of_big_row - none_row_val
                diff = row_val - none_row_val
                extra = f''
                if diff > 0:
                    fraction = max(7, int(diff/max_diff * 40)) 
                    extra = f'\\cellcolor{{blue!{fraction}}}'
                elif diff < 0:
                    extra = f'\\cellcolor{{red!6}}'
                if row == none_row:
                    extra = f'\\cellcolor{{gray!10}}'

                new_row_string += f' & {extra}{row_val}'
            else:
                new_row_string += f' & '


            # none_row_val = table[none_row, col]
            # maximum_of_big_row = np.max(table[none_row:none_row+n_feats, col])
            # if row!= none_row:
            #     row_val = table[row, col] - table[none_row, col]
            #     actual_val = table[row,col]
            #     if row_val > 0 and maximum_of_big_row-none_row_val != 0:
            #         fraction_of_max = max(7, int((actual_val-none_row_val)/(maximum_of_big_row-none_row_val) * 40))
            #         new_row_string += f'& \\cellcolor{{blue!{fraction_of_max}}}{actual_val:.2f}'  
            #     elif row_val < 0 and maximum_of_big_row-none_row_val != 0:
            #         fraction_of_max = int(abs((actual_val-none_row_val)/(maximum_of_big_row-none_row_val) * 1))
            #         new_row_string += f'& \\cellcolor{{red!6}}{actual_val:.2f}'
            #     else:
            #         new_row_string += f'& {actual_val:.2f} '
            # else:
            #     row_val = none_row_val
            #     new_row_string += f'& \\cellcolor{{gray!20}}{row_val:.2f} '

        new_table.append(new_row_string+' \\\\')
    new_table.append('\\hline')
        
print_table(new_table)







