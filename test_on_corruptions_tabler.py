import numpy as np

def print_table(table):
    for row in table:
        print(row)
    print('\n\n')

model_name = 'wrn' # 'wrn' or 'resnet18'

CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression', 
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

for model_type in ['normal']: # 'Stacked', 
    data = {}
    for c in CORRUPTIONS:
        data[c] = {}
    
    epsilon = 0

    if model_name == 'wrn':
        epsilonz = list(range(0, 14, 2)) 
        dt = 2
        filename = f'./logs/wrn_{model_type}_corruptions.log' 
    else:
        if model_type == 'Stacked':
            epsilonz = list(range(0, 14, 2)) 
            dt = 2
        else:
            epsilonz = list(range(0, 13, 1))
            dt = 1
        filename = f'./logs/{model_type}_corruptions.log'        

    corruption_idx = 0
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:                
            if 'Test Error' in line:
                data[CORRUPTIONS[corruption_idx]][epsilon] = 100.0 - float(line.split(' ')[-1])
                corruption_idx += 1
            if 'saturate' in line:
                epsilon += dt
                corruption_idx = 0

    full_str = 'corruption & non-robust '
    for eps in epsilonz[1:]:
        full_str += f'& {eps} '
    print(f'{full_str}\\\\')
    print(f'\\hline')

    sum_of_rows_dict = {}
    n_rows = len(data.items())
    for corruption, row_dict in data.items():
        full_str = f'{corruption} '
        for idx, (epsilon, acc) in enumerate(row_dict.items()):
            if epsilon in sum_of_rows_dict:
                sum_of_rows_dict[epsilon] += acc
            else:
                sum_of_rows_dict[epsilon] = acc
            # if idx == 0: 
            #     non_robust_acc = acc
            #     full_str += f'& {round(acc,2)} '
            # else:
            #     diff = round(acc-non_robust_acc,2)
            #     if diff > 0:
            #         full_str += f'& \\blue{{+{diff}}} '
            #     elif diff < 0:
            #         full_str += f'& \\red{{{diff}}} '
            #     elif diff == 0:
            #         full_str += f'& {diff} '
            if acc == np.sort(list(row_dict.values()))[-1]:
                full_str += f'& \\textbf{{{round(acc,2)}}} '
            elif acc == np.sort(list(row_dict.values()))[-2]:
                full_str += f'& \\underline{{{round(acc,2)}}} '
            else:
                full_str += f'& {round(acc,2)} '

        for c_idx in range(len(full_str)):
            if full_str[c_idx] == '_':
                full_str = f'{full_str[:c_idx]} {full_str[c_idx+1:]}'
        print(f'{full_str}\\\\')

    print(f'\\hline')
    print(f'\\hline')
    full_str = 'Avg. '
    for (epsilon, sum_of_accs) in sum_of_rows_dict.items():
        full_str += f'& {round(sum_of_accs/n_rows, 2)} '
    print(full_str)
    print(f'\n\n\n')

    