#!/usr/bin/env python3
"""
Created on 01.12.2021
@author: Mateusz Fortunka

"""

import sys
import pandas as pd

if __name__ == "__main__":
	if len(sys.argv)!=3:
		sys.exit("Wrong number of arguments given. Correct syntax: pdb_translate.py file.pdb promals_hits")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: pdb_translate file.pdb promals_hits")
		promals_hits=sys.argv[2]

#file_pdb = "/home/users/mfortunka/Documents/docking/dist_program/interface_1h3e_pymol2_AE.pdb"
#promals_hits = "/home/users/mfortunka/Documents/docking/dist_program/promals_output_allhits.txt"
#line = pdb_lines[0]

def aa_code_translate(aa, direction = "1-3" ):
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
		for i in range(len(aa)):
			y += d_rev[aa[i:i+1]]
		return y
    
#this function adds spaces in front of a number which is previously converted to string
#it preserves proper layout of the columns in processed pdb file
def add_spaces(integer, entire_len):
    no_of_spaces=entire_len-len(str(integer))
    newstring= ' ' * no_of_spaces + str(integer)
    return newstring

#this function replaces fragment of a string with a substring in exact place
def str_replace(instr, start, end, substr):
    outstr = instr[:start] + instr[start:].replace(instr[start:end], substr, 1)
    return outstr

hits_tab = pd.read_csv(promals_hits, sep="\t")
#pdb_tab = pd.read_csv(file_pdb, sep="\t")

pdb = open(file_pdb, "r")
pdb_lines = pdb.readlines()
pdb.close()

out = file_pdb[0:-4]+"_translated"+".pdb" 
outfile = open(out, "w")


for line in pdb_lines:
	if line[0:6] == 'HETATM' or line[0:4] == "ATOM":
		i = line.split()
		row = hits_tab[hits_tab["homologic_prot_no"] == int(i[5])]
		if len(row) != 0:
			
			aa_nm1 = row['target_prot_aa'].tolist()
			aa_nm3 = aa_code_translate(str(aa_nm1[0]), "1-3" )
			aa_no = str(row['target_prot_no'].tolist()[0])
			line=str_replace(line, 17, 20, aa_nm3)
			line=str_replace(line, 23, 26, add_spaces(aa_no, 3))
			outfile.write(line)

outfile.write("END\n")
outfile.close()
