#!/usr/bin/env python3
"""
Created on 01.12.2021
@author: Mateusz Fortunka

"""
from copy import deepcopy
import sys
import pandas as pd
import re

if __name__ == "__main__":
    if len(sys.argv) == 3:
        dist_dat = sys.argv[1]
        key = sys.argv[2]
        numbers = ""
        # cutoff = float(sys.argv[2])
    elif len(sys.argv) == 4:
        dist_dat = sys.argv[1]
        key = sys.argv[2]
        numbers = sys.argv[3]
    else:
        sys.exit(
            "Wrong number of arguments given. Correct syntax: contact_sift.py output_dist.dat key_to_choose_structures (all/best/range) number(s) (inclusively and with no space or brackets).")

if key == "best":
    range_ = range(1, int(numbers) + 1)
elif key == "range":
    r = re.split(",|-|;|_|", numbers)
    range_ = range(int(r[0]), int(r[1]) + 1)


def find_lno_str(s: str, lines: list):
    for i, v in enumerate(lines):
        if s in v:
            return i


dist_file = open(dist_dat, "r")
dist_lines = dist_file.readlines()
dist_file.close()
contact_counter = find_lno_str("END", dist_lines) - find_lno_str("START", dist_lines)
models = {}
contacts = []
li = 0
example_name = ""
while li < len(dist_lines):
    l = dist_lines[li]
    if l[:5] == "START":
        name = l.split()[1]
        if key != "all":
            act_no = int(name.split("_")[1].rstrip(".pdb"))
            if act_no not in range_:
                li += contact_counter + 1
    elif l[:3] == "END":
        models[name] = (deepcopy(contacts))
        contacts = []
        if example_name == "":
            example_name = deepcopy(name)
    elif l != "" and l != "\n" and l[0] != "/":
        line = l.split()
        line[6] = float(line[6])
        contacts.append(line)
    li += 1

out = key + numbers + "_contacts.dat"
outfile = open(out, "w")
map_ = pd.DataFrame(models[example_name])
map_[6] = 0

for m in models.keys():
    M = pd.DataFrame(models[m])
    median = pd.DataFrame.median(M[6])
    for idx, row in M.iterrows():
        if row[6] <= median:
            map_.loc[idx, 6] += 1

map_ = map_.sort_values(by=6, ascending=False)
for idx, row in map_.iterrows():
    row[6] = str(row[6])
    outfile.write("\t".join(row.tolist()) + "\n")
outfile.close()
