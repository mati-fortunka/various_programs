# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#import pandas as pd

fname = '1/contact.dat' 
my_data = np.genfromtxt(fname, skip_header=1)

out = fname[0:-4]+"_list"+".dat" 
outfile = open(out, "w")

for j in range(len(my_data)):
    line = f"{j}: "
    for i, v in enumerate(my_data[j]):
            if v!=0:
                line += f'{i}, '
    outfile.write(f"{line}\n")
outfile.close()
