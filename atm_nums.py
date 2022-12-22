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
	
#this function adds spaces in front of a number which is previously converted to string
#it preserves proper layout of the columns in processed pdb file
def add_spaces(integer, entire_len):
    no_of_spaces=entire_len-len(str(integer))
    newstring= ' ' * no_of_spaces + str(integer)
    return newstring

infile = open(file_pdb, "r")
conv_pdb = file_pdb[0:-4]+"_n"+".gro" 
outfile = open(conv_pdb, "w")

atm_no = 2474
start_no = 1

flag = 0
idx = 0
for line in infile:
	try:
		atm = line[16:20]
		if flag == 0 and int(atm) == atm_no:
			line=str_replace(line, 16, 20, add_spaces(start_no+idx, 4))
			flag = 1
			idx += 1
		else:
			line=str_replace(line, 16, 20, add_spaces(start_no+idx, 4))
			idx += 1
	except:
		pass
	outfile.write(line)
        
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_pdb} file.")


