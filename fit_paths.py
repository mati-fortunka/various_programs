# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:50:37 2021
Updated 12.06.2023!
@author: Forty
"""

import os
import sys
import glob

if __name__ == "__main__":
    if len(sys.argv)!=2:
        sys.exit("Wrong number of arguments given. Correct syntax: fit_paths.py path_to_folder")
    else:
        folder = sys.argv[1]
folder = "/home/matifortunka/Documents/JS/publication_YARS/apoenzyme/2pid_mut/vmd_dd/0"

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
    print(f"Processing {i.split('/')[-1]}")
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
    nodes_opt = all_lines[6].split(',')
    nodes_opt.pop()
    nodes_set = set(map(int, nodes_opt))
    for l in all_lines[7:]:
        if l[0:6]=="Number":
            outfile.write(l)
            break        
        else:
            new_nodes = l.split(',')
            new_nodes.pop()
            new_nodes = set(map(int, new_nodes))
            if new_nodes.issubset(nodes_set)==False:
                outfile.write(l)
                nodes_set.update(new_nodes)
    outfile.close()
