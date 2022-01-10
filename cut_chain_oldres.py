#!/usr/bin/env python3
"""
Created on 
Finished on 
@author: Mateusz Fortunka

"""

import sys

if __name__ == "__main__":
	if len(sys.argv) < 3:
		sys.exit("Wrong number of arguments given. Correct syntax: cut_chain.py file.pdb chain1_id chain2_id chain3_id ...")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: make_one_chain.py file.pdb")
		chain=[sys.argv[i] for i in range(2,len(sys.argv))]

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
conv_pdb = file_pdb[0:-4]+"_"+ ''.join(chain) +".pdb" 
outfile = open(conv_pdb, "w")
atom_no=1
res_no=0
old_line=""
res_name='   0'
for line in infile:
    
    if line[0:3] == 'TER':
        if old_line[0:3] != 'TER' or old_line=="":
            old_line=line
            outfile.write(line)
        continue
        
    elif line[0:3] == 'END':
        outfile.write('END')
        infile.close()
        outfile.close()
        sys.exit(f'Succesfully written output to the {conv_pdb} file.')

    elif line[0:6] == 'HETATM' or line[0:4] == "ATOM":
        if line[17:20] == 'HOH':
            continue
        elif line[21:22] in chain:
            #line=str_replace(line, 77, 81, '   ')    
            line=str_replace(line, 0, 6, 'ATOM  ')
            #line=str_replace(line, 21, 22, 'A')
            if line[22:26]!=res_name: 
                res_no+=1
            res_name=line[22:26]
            line=str_replace(line, 5, 11, add_spaces(atom_no, 6))
            #line=str_replace(line, 22, 26, add_spaces(res_no, 4))
            atom_no+=1
            outfile.write(line)
            old_line=line

