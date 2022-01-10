#!/usr/bin/env python3
"""
Created on 05.11.2021
@author: Mateusz Fortunka

Python version >= 3.6 needed because of fstring usage.

Usage: 
promals_translate_info.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt

This program is somehow similar to promals_translate.py, but gives human-readable output very useful in manual verification and analysis of aminoacids translated from the homologic protein to the target one. It takes two arguments: 
 - promals_output_target_vs_homologic.txt: Output from "promals_hits.py" (Promals3D output visualised in one line -> copied to the .txt file -> used as an imput to promals_hits.py"). Four columns are necessary: target_prot_no, target_prot_aa, homologic_prot_no and consensus_symbol.
 - aas_list_from_homologic.txt: List of aminoacids from homologic protein e.g. which are in contact with docked RNA/ligand. Two columns are necessary: "res_no" and "chain". It can be prepared with "list_res+aa_code_conv.py" program.

Output gives one letter codes and numbers of aminoacids taken from "promals_output_target_vs_homologic.txt" file and consensus symbol:

conserved amino acid residues: bold and uppercase letters (such as G)
aliphatic residues (I, V, L): l 
aromatic residues (Y, H, W, F): @ 
hydrophobic residues (W, F, Y, M, L, I, V, A, C, T, H): h
alcohol residues (S, T): o
polar residues (D, E, H, K, N, Q, R, S, T): p
tiny residues (A, G, C, S): t
small residues (A, G, C, S, V, N, D, T, P): s
bulky residues (E, F, I, K, L, M, Q, R, W, Y): b
positively charged residues (K, R, H): +
negatively charged residues (D, E): -
charged (D, E, K, R, H): c

"""

import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv)!=3:
        sys.exit("Wrong number of arguments given. Correct syntax: promals_translate_info.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt")
    else:
        hits = sys.argv[1]
        aa_list = sys.argv[2]

hits_tab = pd.read_csv(hits, sep="\t")
aa_tab = pd.read_csv(aa_list, sep="\t")
aa_tab["res_no"] = aa_tab["res_no"].astype(int)
aa_tab = aa_tab.sort_values(["res_no", "chain"])

#out = hits[0:-4]+aa_list[0:-4]+"info.txt"
out = "translated_info.txt"
v_old="-1"
outfile = open(out, "w")
outfile.write("res_no\tchain\ttarget_prot_no\ttarget_prot_aa\thomologic_prot_no\thomologic_prot_aa\tconsensus_symbol\n")

for i, v in enumerate(hits_tab["homologic_prot_no"].tolist()):
    if v == v_old: continue
    v_old = v
    aa_hit = aa_tab[(aa_tab['res_no'] == int(v))]
    if len(aa_hit) > 0:
        line = ""
        rline = f"{hits_tab['target_prot_no'][i]}\t{hits_tab['target_prot_aa'][i]}\t{hits_tab['homologic_prot_no'][i]}\t{hits_tab['homologic_prot_aa'][i]}\t{hits_tab['consensus_symbol'][i]}\n"
        for c in aa_hit['chain']:
            line += f"{v}\t{c}\t"+rline

        outfile.write(line)

outfile.close()
sys.exit(f"Output succesfully written to {out} file.")

