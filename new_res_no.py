#!/usr/bin/env python3
"""
Created on 8.07.2022
@author: Mateusz Fortunka
"""

import sys

#resid = "AA"
syntax = 'new_res_no.py file.gro file.itp resid'

if __name__ == "__main__":
	if len(sys.argv)!=4:
		sys.exit(f"Wrong number of arguments given. Correct syntax: {syntax}")
	else:
		file_gro = sys.argv[1]
		file_itp = sys.argv[2]
		resid = sys.argv[2]
		if file_gro[-4:] != '.gro':
			print(f"Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: {syntax}")

#this function replaces fragment of a string with a substring in exact place
def str_replace(instr, start, end, substr):
	outstr = instr[:start] + instr[start:].replace(instr[start:end], substr, 1)
	return outstr

#this function adds spaces in front of a number which is previously converted to string
#it preserves proper layout of the columns in processed gro file
def add_spaces(integer, entire_len):
    no_of_spaces=entire_len-len(str(integer))
    newstring= ' ' * no_of_spaces + str(integer)
    return newstring

atom_seq = []

itp = open(file_itp, "r")
itp_lines = itp.readlines()
flag = 0
for i, v in enumerate(itp_lines):
	if "[ atoms ]" in v :
		flag = 1
		for j in itp_lines[i+1:]:
			j = j.split()
			if len(j)>4 and j[0] != ";" and j[3] == resid:
				atom_seq.append(j[4])
			if j == "\n":
				break
	elif flag == 1:
		break

infile = open(file_gro, "r")
conv_gro = file_gro[0:-4]+"_n"+".gro" 
outfile = open(conv_gro, "w")


no_occurence = [0] * len(atom_seq)
atoms = dict(zip(atom_seq, no_occurence))

for idx, line in enumerate(infile):
	l = line.split()
	if len(l)>4 and resid in l[3]:
		atoms[l[2]] +=1
		no = atoms[l[2]]
		if l[4] != no:
			line = str_replace(line, 22, 26, add_spaces(no, 4))
	outfile.write(line)

        
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_gro} file.")


