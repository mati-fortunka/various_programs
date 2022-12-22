#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 17:46:52 2022

@author: mfortunka
"""
import numpy as np
import pandas as pd

chains = "AC"

c_dict = {}

def add(cont, occupancy):
    if cont not in c_dict.keys():
        c_dict[cont] = []
    c_dict[cont].append(occupancy)
    
for no in range(1,8):

    fname = f'{no}/hbonds_{chains}.dat'     
    file = open(fname, "r")
    lines = file.readlines()
    file.close()
    table = []
    lines = lines[2:]

    for i, v in enumerate(lines):
        lines[i] = v.replace('%','').replace('-Main','').replace('-Side','')
        line = lines[i].split()
        if chains == 'AC':
            contact = f"{line[0]}-{line[1]}"
            value = float(line[2])

        else:
            if "ATP" not in line[0] and "tyr" not in line[0]:
                contact = f"{line[0]}"
            else:
                contact = f"{line[1]}"
            value = float(line[2])
        table.append([contact, value])    
    
    df = pd.DataFrame(table, columns=('contact', 'value'))        
    df = df.groupby(by='contact', as_index = False).sum()
    
    #sys.exit("Error message")
    #df[df['value'] > 100]['value'] = 100
    for i in list(df.index):
        v = df['value'][i]
        if v > 100:
            v = 100
        add(df['contact'][i], v)
        
out = f"hbonds_{chains}_list.dat" 
outfile = open(out, "w")

i=0
keys_list = list(c_dict.keys())
while i < len(keys_list):
    v = keys_list[i]
    if sum(c_dict[v]) < 30:
        del c_dict[v]
    i+=1

for i in c_dict.keys():    
    n = 7 - len(c_dict[i])
    if n > 0:
        for j in range(n):
            c_dict[i].append(0)

    outfile.write(f'{i}\t{np.mean(c_dict[i])}\t{np.std(c_dict[i])}\n')

outfile.close()


