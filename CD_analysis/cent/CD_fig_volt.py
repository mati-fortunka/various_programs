# -*- coding: utf-8 -*-
"""
Created 8.11.2021
Based on program from ug 24 14:09:38 2020

@author: Forty
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Wrong number of arguments given. Correct syntax: CD_fig_volt.py path_to_spectrum name")
    else:
        spectrum = sys.argv[1]
        name = sys.argv[2]

deg = u"\N{DEGREE SIGN}"
col=["#a20021","#4f5d2f","#fdb833","#ff8595","#3d315b","#310a31","#08080c"]
col2=['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']

path = spectrum[:-4]
XY_spectrum=np.genfromtxt(spectrum, dtype='float64', skip_header=19, usecols = (0,2))


title="CD " + name + "- napięcie fotopowielacza"
plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')    
plt.plot(XY_spectrum[:,0], XY_spectrum[:,1], linewidth=2, color=col[4], label=None)
plt.xlabel('Długość fali [nm]')
plt.ylabel('Napięcie fotopowielacza [V]')
plt.title(title)
#plt.axis([0, 210 , 0, 0.58])
#plt.axis([0, (i+10)*dt , 0, r+0.5])
plt.grid()
plt.savefig(path +'_V.png')
plt.close()

