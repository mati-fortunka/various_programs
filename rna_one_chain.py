#!/usr/bin/env python3
"""
Created on 
Finished on 
@author: Mateusz Fortunka

"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: rna_one_chain.py file.pdb")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: rna_one_chain.py file.pdb")

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
conv_pdb = file_pdb[0:-4]+"_ronech"+".pdb" 
outfile = open(conv_pdb, "w")
atom_no=1
res_no=0

res_name='   0'
for line in infile:
    
    if line[0:3] == 'TER':
        continue
        
    elif line[0:3] == 'END':
        outfile.write('TER\n')
        outfile.write('END')
        infile.close()
        outfile.close()
        sys.exit(f'Succesfully written output to the {conv_pdb} file.')

    elif line[0:6] == 'HETATM' or line[0:4] == "ATOM":
        #if line[17:20] == 'HOH' or line[77] == 'H':
        if line[17:20] == 'HOH':
            continue
        if line[17:20] == '1MA':
            line=str_replace(line, 17, 20, ' RA')
        if line[17:20] == 'PSU' or line[17:20] == '5MU':
            line=str_replace(line, 17, 20, ' RU')
        line=str_replace(line, 0, 6, 'ATOM  ')
        line=str_replace(line, 21, 22, 'A')
        line=str_replace(line, 77, 81, '   ')
        if line[22:26]!=res_name: 
            res_no+=1
        res_name=line[22:26]
        line=str_replace(line, 5, 11, add_spaces(atom_no, 6))
        line=str_replace(line, 22, 26, add_spaces(res_no, 4))
        atom_no+=1
        outfile.write(line+'\n')


