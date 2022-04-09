#!/usr/bin/env python3
"""
Created on 01.12.2021
@author: Mateusz Fortunka

"""
from copy import deepcopy
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Wrong number of arguments given. Correct syntax: model_sift.py output_dist.dat cutoff")
    else:
        dist_dat = sys.argv[1]
        cutoff = float(sys.argv[2])

# file_pdb = "/home/users/mfortunka/Documents/docking/dist_program/interface_1h3e_pymol2_AE.pdb"
# promals_hits = "/home/users/mfortunka/Documents/docking/dist_program/promals_output_allhits.txt"
# line = pdb_lines[0]

dist_file = open(dist_dat, "r")
dist_file.readline()
dist_file.readline()
dist_lines = dist_file.readlines()
dist_file.close()

models = {}
contacts = []
for l in dist_lines:
    if l[:5] == "START":
        name = l.split("\t")[1]
    elif l[:3] == "END":
        models[name] = (deepcopy(contacts))
        contacts = []
        continue
    elif l == "" or l == "\n":
        continue
    else:
        line = l.split("\t")
        line[6] = float(line[6])
        contacts.append(line)

out = "models_sifted.dat"
outfile = open(out, "w")

for m in models.keys():
    M = pd.DataFrame(models[m])
    avg = sum(M[6]) / len(M[6])
    if avg <= cutoff:
        outfile.write(f"START\t{m}\n")
        for index, row in M.iterrows():
            if row[6] <= cutoff:
                row[6] = str(row[6])
                outfile.write("\t".join(row.tolist()) + "\n")
        outfile.write("END\n")

outfile.close()
