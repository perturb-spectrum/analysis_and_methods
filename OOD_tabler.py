import numpy as np

def print_table(table):
    for row in table:
        print(row)
    print('\n\n')

epsilonz = list(range(0, 13, 1)) 
deltaz = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

auroc_table = -1*np.ones((len(deltaz), len(epsilonz))) 
tnr_table = -1*np.ones((len(deltaz), len(epsilonz))) 

for delta in deltaz:
    filename = f'./logs/scores_when_eval_on_{int(delta)}.log'
    with open(filename) as file:
        lines = file.readlines()
        epsilon = 0
        for line in lines:
            if 'AUROC' in line:
                auroc = float(line.split(',')[-2].split(' ')[-1])
                tnr = float(line.split(',')[-1].split(' ')[-1])*100.0
                auroc_table[delta, epsilon] = auroc
                tnr_table[delta, epsilon] = tnr
                epsilon += 1

T = 1
for row_idx in range(len(auroc_table)):
    row_str = f'{row_idx} & {T} & '
    for col_idx in range(len(auroc_table[0])):
        row_str += f'{auroc_table[row_idx, col_idx]} & '

        