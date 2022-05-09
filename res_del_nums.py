#!/usr/bin/env python3
"""
Created on 9.04.2022
@author: Mateusz Fortunka
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: res_del_nums.py file.pdb")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.pdb':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: smog_conv.py file.pdb")

#this function replaces fragment of a string with a substring in exact place
def str_replace(instr, start, end, substr):
	outstr = instr[:start] + instr[start:].replace(instr[start:end], substr, 1)
	return outstr

infile = open(file_pdb, "r")
conv_pdb = file_pdb[0:-4]+"_delnum"+".pdb" 
outfile = open(conv_pdb, "w")
num = 264
for line in infile:
	if line[21:22] == 'C' and int(line[23:26]) == num+1:
		line=str_replace(line, 23, 26, str(num))
	outfile.write(line)
        
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_pdb} file.")


