#!/usr/bin/env python3
"""
Created on 04.11.2021
@author: Mateusz Fortunka

Python version >= 3.6 needed because of fstring usage.


This program takes Promals3D output visualised in one line on site and then copied to the .txt file. It produces two files *_besthits.txt and *_allhits.txt with aligned aminoacids. In the first one only fully conserved aminoacids are present. In the second all partial hits are also included. The last column specify the type of alignment:

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

If you want to use the output in promals_translate.py program remember to specify target protein in Promals3D as first, and homologic as second. It is important!

Remember that Promals3D do not understand non canonical aminoacid names (e.g. HSE, HSD, GLUP), so change them before running alignment.

Usage:
./promals_hits.py promals_oneline_output.txt no_of_first_aa_in_target_prot no_of_first_aa_in_homologic_prot

Command is self explanatory, but I will give more details.
 - promals_oneline_output.txt: Promals3D output visualised on server in one line and copied to a .txt file. It is essential for good operation of this script to have every aligned sequence and consensus symbols in separated lines (one each).
 - no_of_first_aa_in_target_prot: Number of the first aminoacid in the target protein. Oftentimes crystal structure of a protein does not include a couple of starting aminoacids. To keep good numbering (in agreement with the order specified in the representative genome translated to aa, but practically with pdb file) this script needs the number of the first aminoacid.
 - no_of_first_aa_in_homologic_prot:  Same as above, but in the homologic (second) protein.

"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=4:
		sys.exit("Wrong number of arguments given. Correct syntax: promals_hits.py promals_oneline_output.txt no_of_first_aa_in_target_prot no_of_first_aa_in_homologic_prot")
	else:
		file_hits=sys.argv[1]
		drift1=int(sys.argv[2])+1
		drift2=int(sys.argv[3])+1

infile = open(file_hits, "r")
if file_hits[-4:] == ".txt":
	out1 = file_hits[0:-4]+"_besthits"+".txt"
	out2 = file_hits[0:-4]+"_allhits"+".txt"
else:
	out1 = file_hits+"_besthits"+".txt" 
	out2 = file_hits+"_allhits"+".txt"
conserv = infile.readline()
first = infile.readline()
second = infile.readline()
consens = infile.readline()
infile.close()

end_idx = len(consens)
start_idx = consens.find(".")
conserv = conserv[start_idx:]
first_prot = first[start_idx : end_idx]
second_prot = second[start_idx : end_idx]
consens = consens[start_idx : ]

def find(s, ch):
	return [str(i) for i, ltr in enumerate(s) if ltr == ch]

conserv_list = find(conserv, "9")
out_list = find(consens, ".")
consens_list = [str(i) for i in range(len(consens)) if str(i) not in out_list]

best = ""
for i in conserv_list:
	i = int(i)
	gaps1 = first_prot[:i].count('-')
	gaps2 = second_prot[:i].count('-')
	hit = f"{i+drift1-gaps1}\t{first_prot[i]}\t{i+drift2-gaps2}\t{second_prot[i]}\t{consens[i]}\n"
	best += hit
all_hits = ""
for i in consens_list:
	i = int(i)
	gaps1 = first_prot[:i].count('-')
	gaps2 = second_prot[:i].count('-')
	hit = f"{i+drift1-gaps1}\t{first_prot[i]}\t{i+drift2-gaps2}\t{second_prot[i]}\t{consens[i]}\n"
	all_hits += hit

outfile = open(out1, "w")
outfile.write(f"target_prot_no\ttarget_prot_aa\thomologic_prot_no\thomologic_prot_aa\tconsensus_symbol\n")
outfile.write(best)
outfile.close()

outfile = open(out2, "w")
outfile.write(f"target_prot_no\ttarget_prot_aa\thomologic_prot_no\thomologic_prot_aa\tconsensus_symbol\n")
outfile.write(all_hits)
outfile.close()

sys.exit(f"Output succesfully written to {out1} and {out2} files.")


