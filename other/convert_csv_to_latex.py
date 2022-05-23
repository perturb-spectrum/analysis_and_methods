import numpy as np

def print_table(table):
    for row in table:
        print(row)
    print('\n\n')

filename = f'./combined.csv'
table = []
with open(filename) as file:
    lines = file.readlines()
    for line in lines:
        table.append(line.replace(',', ' & ')+'\\\\')

print_table(table)