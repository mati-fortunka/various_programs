#!/usr/bin/env python3
"""
Created on 9.04.2022
@author: Mateusz Fortunka
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=3:
		sys.exit("Wrong number of arguments given. Correct syntax: atm_seq.py file.gro file.itp")
	else:
		file_pdb = sys.argv[1]
		file_itp = sys.argv[2]
		if file_pdb[-4:] != '.gro':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: atm_seq.py file.gro file.itp")

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

atom_seq = []
resid = "PAA"

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

infile = open(file_pdb, "r")
conv_pdb = file_pdb[0:-4]+"_n"+".gro" 
outfile = open(conv_pdb, "w")

old_lines = []
old_l = [""]
no = 0
saved_lines = []
for idx, line in enumerate(infile):
	l = line.split()
	if resid in l[0]:
		resno = l[0].replace(resid, "")
		if resno not in old_l[0]:
			start_idx = idx-1
			no = 0
		if l[1] == atom_seq[no]:
			if l[2] != start_idx+no:
				line = str_replace(line, 15, 20, add_spaces(start_idx+no, 5))
			outfile.write(line)
			no += 1
		else:
			if not saved_lines:

				saved_lines.append(line)
			else:
				saved_lines.append(line)

				s = 0
				while 1:
					if s < len(saved_lines):
						if saved_lines[s].split()[1] == atom_seq[no]:
							if l[2] != start_idx+no:
								line = str_replace(saved_lines[s], 15, 20, add_spaces(start_idx+no, 5))
							outfile.write(line)
							del saved_lines[s]
							no += 1

						else:
							s += 1
					else:
						break
	else:
		outfile.write(line)
	old_l = line.split()
        
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_pdb} file.")


