# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:50:37 2021

@author: Forty
"""

import os
import sys
import glob

if __name__ == "__main__":
    if len(sys.argv)!=2:
        sys.exit("Wrong number of arguments given. Correct syntax: psummary2.py path_to_folder")
    else:
        folder = sys.argv[1]
#folder = "/home/users/mfortunka/Documents/MD/2pid/apo/1/"

path = folder + "/fit_paths/"
files = [f for f in glob.glob(folder + "/*.out.out", recursive=True)]
try:
    os.mkdir(path)
except FileExistsError:
    pass

for i in files:
    infile = open(i, "r")
    all_lines = infile.readlines()
    infile.close()
    outfile = open(path+i.lstrip(folder), "w")
    print()
    #outfile.write(str(all_lines[:3]))
    outfile.write(all_lines[0])
    outfile.write(all_lines[1])
    outfile.write(all_lines[2])
    start = int(all_lines[3][-5:-2])
    end = int(all_lines[4][-5:-2])
    if start>end: start, end = end, start
    outfile.write(f"The sources are: {start} \nThe targets are: {end} \n") 
    outfile.write(all_lines[5])
    outfile.write(all_lines[6])
    nodes_set = set(map(int, all_lines[6][:-7].split(',')))
    for l in all_lines[7:]:
        if l[0:6]=="Number":
            outfile.write(l)
            break        
        else:
            nodes = l[:-7].split(',')
            if set(nodes).issubset(nodes_set)==False:
                outfile.write(l)
                nodes_set.update(nodes)
    outfile.close()
