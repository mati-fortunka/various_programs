#!/usr/bin/env python3

import sys, os
import fileinput

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Wrong number of arguments given. Correct syntax: exchange_chain.py directory chain1_id chain2_id")
    else:
        directory = sys.argv[1]
        ch_1 = sys.argv[2]
        ch_2 = sys.argv[3]

files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)

files.remove("trna_1h3e.pdb")
# files.sort()

for f in files:
    flag = None
    ter = None
    with open(f, 'r') as pdbf:
        lines = pdbf.readlines()
    for i, l in enumerate(lines):
        split = l.split()
        if split[0] == "TER":
            ter = "TER"
        elif split[0] == "ATOM" or split[0] == "HETATM":
            if l[21] == ch_1:
                lines[i] = l[0:21] + ch_2 + l[22:]
                if flag is None:
                    flag = "changed"
                    ter = None
        if ter == "TER" and flag == "changed":
            break
    with open(f, "w") as pdbf:
        pdbf.writelines(lines)
"""

    for l in fileinput.FileInput(f, inplace=1):
        if l[0:4] == "ATOM" or l[0:6] == "HETATM":
            if l[21] == ch_1:
                l[21] = ch_2
            elif l[21] == ch_2:
                l = ch_1
            #l = newline.rsplit("\n")
        print(l)
"""
