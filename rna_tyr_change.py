#!/usr/bin/env python3

import sys, os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Wrong number of arguments given. Correct syntax: rna_tyr_change.py file flag (rna/dna)")
    else:
        f = sys.argv[1]
        flag = sys.argv[2]


with open(f, 'r') as read:
    lines = read.readlines()

for i, l in enumerate(lines):
    split = l.split()
    if split[0] == "ATOM" or split[0] == "HETATM":
            if flag == "rna":
                lines[i] = lines[i].replace(" RA", "ADE")
                lines[i] = lines[i].replace(" RC", "CYT")
                lines[i] = lines[i].replace(" RG", "GUA")
                lines[i] = lines[i].replace(" RU", "URA")
                lines[i] = lines[i].replace("tyr", "TYR")
            elif flag == "dna":
                lines[i] = lines[i].replace("ADE", " RA")
                lines[i] = lines[i].replace("CYT", " RC")
                lines[i] = lines[i].replace("GUA", " RG")
                lines[i] = lines[i].replace("URA", " RU")
                if int(split[5]) >= 476:
                    lines[i] = lines[i].replace("TYR", "tyr")
            else:
                print("Type proper flag - rna for RX -> NUC (e.g. RC -> CYT) or dna for NUC -> RX (e.g. GUA -> RG).")

with open(f, "w") as write:
    write.writelines(lines)


