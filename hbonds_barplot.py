#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 01:12:28 2022

@author: mfortunka
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv)!=2:
        sys.exit("Wrong number of arguments given. Correct syntax: vmd_hbonds.py contacting_chains (e.g. AB)")
    else:
        chains = sys.argv[1]

#chains = "CG"
fname1 = f'../ab72/bonds/hbonds_{chains}_list.dat'     
fname2 = f'../mut_ab72/bonds/hbonds_{chains}_list.dat'     

data = pd.read_csv(fname1, names=('contact', 'occurence', 'st. dev.'), delimiter=('\t'))
data_mut = pd.read_csv(fname2, names=('contact', 'occurence', 'st. dev.'), delimiter=('\t'))

for i, v in enumerate(data['contact']):
    if data['occurence'][i] < data['st. dev.'][i]:
        data.loc[i,'occurence'] = data.loc[i,'st. dev.'] 
    if v not in list(data_mut['contact']):
        df = pd.DataFrame({'contact': [v], 'occurence': [0], 'st. dev.': [0]})
        data_mut = data_mut.append(df, ignore_index=True)
        
for j, v in enumerate(data_mut['contact']):
    if data_mut['occurence'][j] < data_mut['st. dev.'][j]:
        data_mut.loc[j, 'occurence'] = data_mut.loc[j, 'st. dev.'] 
    if v not in list(data['contact']):
        df = pd.DataFrame({'contact': [v], 'occurence': [0], 'st. dev.': [0]})
        data = data.append(df, ignore_index=True)

data=data.sort_values('contact')
data_mut=data_mut.sort_values('contact')

#errors = data['st. dev.'].merge(mut_data['st. dev.'])

#df = [data[], data_mut[]]
#data.plot.bar(x='contact', y='occurence', yerr='st. dev.', figsize=(14,10))

#X = np.arange(4)
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(data['contact'], data['occurence'], color = 'b', width = 0.25)
#ax.bar(data['contact'], data_mut['occurence'], color = 'g', width = 0.25)

x_pos = np.array([i for i in range(0,2*len(data['contact']),2)])
x_pos2 = x_pos+0.8


fig, ax = plt.subplots(figsize=(10, 8))
ax.bar(x_pos, data['occurence'], yerr=data['st. dev.'], align='center', alpha=1, ecolor='black', capsize=6)
ax.bar(x_pos2, data_mut['occurence'], yerr=data_mut['st. dev.'], align='center', alpha=0.7, ecolor='black', capsize=6)
#ax.legend(["native", "mutated"], loc='upper center')
ax.legend(["native", "mutated"], loc='best')
ax.set_ylabel('Occurence in all simulations [%]')
ax.set_xticks((x_pos+x_pos2)/2)
ax.set_xticklabels(data['contact'], rotation=45)
ax.set_title('Bonds during MD')
ax.yaxis.grid(True)

plt.savefig(f'hbonds_{chains}.png', dpi=300)
plt.show()

