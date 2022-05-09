#!/usr/bin/env python3
"""
Created on 
Finished on 
@author: Mateusz Fortunka

"""

import sys

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Wrong number of arguments given. Correct syntax: conv_pqr_rna.py file.pdb")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pqr':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: make_one_chain.py file.pqr")


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
conv_pdb = file_pdb[0:-4] +"c.pdb" 
outfile = open(conv_pdb, "w")
for line in infile:

    if line[0:6] == 'HETATM' or line[0:4] == "ATOM":
        if line[17:20] == 'ADE': line=str_replace(line, 17, 20, " RA")
        elif line[17:20] == 'GUA': line=str_replace(line, 17, 20, " RG")
        elif line[17:20] == 'CYT': line=str_replace(line, 17, 20, " RC")
        elif line[17:20] == 'URA': line=str_replace(line, 17, 20, " RU")
        elif line[17:20] == 'GUA': line=str_replace(line, 17, 20, " RG")
        elif line[16:20] == "3TER": line=str_replace(line, 16, 20, "  RA")
        elif line[16:20] == "5TER": line=str_replace(line, 16, 20, "  RG")    
        
    outfile.write(line)
