#!/usr/bin/env python3
"""
Created on Thu Apr 29 11:49:53 2021
Finished on May 6 2021
@author: Mateusz Fortunka

Program converts standard .pdb format into input file to the first
step of SMOG program (https://smog-server.org/cgi-bin/GenTopGro.pl).
It cuts water molecules, hydrogens and all lines and additional
columns not accepted by SMOG.

Program assumes that there are maximum 999999 atoms and 9999 residues
in every chain in input .pdb file, characters from 17-20 in ATOM/HETATM
lines state name of the residue, 22nd character may be a letter
representing residue (when there are more molecules/chains in the system)
and characters from 23 to 26 show number of the residue.

Files used in this program must not have any holes in chains like missing 
residues, because it cause errors in SMOG.

SMOG do not accept files with strange residues, so make sure to delete them
before generating contact map using SMOG.
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: smog_conv.py file.pdb")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: smog_conv.py file.pdb")

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


infile = open(file_pdb, "r")
conv_pdb = file_pdb[0:-4]+"_smogconv"+".pdb" 
outfile = open(conv_pdb, "w")
atom_no=1
res_no=0

res_name='   0'
for line in infile:
    
    if line[0:3] == 'TER':
        res_no=0
        res_name='   0'
        outfile.write('TER\n')
        
    elif line[0:3] == 'END':
        outfile.write('END')
        infile.close()
        outfile.close()
        sys.exit(f'Succesfully written output to the {conv_pdb} file.')
    elif line[0:6] == 'HETATM' or line[0:4] == "ATOM":
        if line[17:20] == 'HOH' or line[77] == 'H':
            continue
        line=str_replace(line, 0, 6, 'ATOM  ')
        line=str_replace(line, 21, 22, ' ')
        line=str_replace(line, 77, 81, '   ')
        if line[22:26]!=res_name: 
            res_no+=1
        res_name=line[22:26]
        line=str_replace(line, 5, 11, add_spaces(atom_no, 6))
        line=str_replace(line, 22, 26, add_spaces(res_no, 4))
        atom_no+=1
        outfile.write(line+'\n')


