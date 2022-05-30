#!/usr/bin/env python3
"""
Created on 9.04.2022
@author: Mateusz Fortunka
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: atm_nums.py file.gro")
	else:
		file_pdb=sys.argv[1]
		if file_pdb[-4:] != '.gro':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: atm_nums.py file.gro")

#this function replaces fragment of a string with a substring in exact place
def str_replace(instr, start, end, substr):
	outstr = instr[:start] + instr[start:].replace(instr[start:end], substr, 1)
	return outstr

infile = open(file_pdb, "r")
conv_pdb = file_pdb[0:-4]+"_n"+".gro" 
outfile = open(conv_pdb, "w")
atm_no = 2472
idx = 0
for line in infile:
	try:
		atm = line[16:20]
		if int(atm) == atm_no:
			line=str_replace(line, 16, 20, str(atm_no+idx))
			idx += 1
	except:
		pass
	outfile.write(line)
        
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_pdb} file.")


