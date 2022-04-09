#!/usr/bin/env python3
"""
Created on 29.06.2021
@author: Mateusz Fortunka

Python version >= 3.6 needed because of fstring usage.

Usage: list_res+aa_code_conv.py file.pdb

This program takes pdb file and gives list of residues as a output. It is printing residue number, its three letter code and chain symbol. Script numbers residues globally and in every chain. It also translates three letter codes to one letter adding column with them.

"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: list_res.py file.pdb")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: list_res+aa_code_conv.py file.pdb")

def aa_code_translate(aa, direction = "3-1" ):
	d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'HSE': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
	d_rev = {v: k for k, v in d.items()}
	if direction == "3-1":
		if len(aa) % 3 != 0: raise ValueError('Input length should be a multiple of three')
		else:
			y = ''
			for i in range(int(len(aa)/3)):
				y += d[aa[3*i:3*i+3]]
		return y
	elif direction == "1-3":
		y = ''
		for i in range(len(x)):
			y += d[x[i:i+1]]
		return y		

infile = open(file_pdb, "r")
out = file_pdb[0:-4]+"_reslist"+".txt" 
outfile = open(out, "w")
outfile.write(f"no_global\tno_in_chain\tres_no\tchain\tthree_letter_aa\tone_letter_aa\n")
i=1
j=1
res_name=""
chain=""
for line in infile:
	if line[0:6] == 'HETATM' or line[0:4] == "ATOM":
		if line[23:26]!=res_name:
			if line[21]!=chain: j=1
			#one_letter_aa = aa_code_translate(line[17:21].replace(' ', ''), "3-1")
			one_letter_aa = aa_code_translate(line[17:20], "3-1")
			newline = str(i) + '\t' + str(j) + '\t' + line[23:26] + '\t' + line[21] + '\t' + line[17:21] + '\t' + one_letter_aa + '\n'
			outfile.write(newline)
			i+=1
			j+=1
		res_name=line[23:26]
		chain = line[21]
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {out} file.")



