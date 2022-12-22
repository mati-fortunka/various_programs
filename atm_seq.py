#!/usr/bin/env python3
"""
version II
Created on 9.07.2022
@author: Mateusz Fortunka
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=4:
		sys.exit("Wrong number of arguments given. Correct syntax: atm_seq.py file.gro file.itp resid")
	else:
		file_pdb = sys.argv[1]
		file_itp = sys.argv[2]
		resid = sys.argv[3]
		if file_pdb[-4:] != '.gro':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: atm_seq.py file.gro file.itp resid")

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
#resid = "AA"

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
old_resno = ""
resno = ""
no = 0
saved_lines = []
for idx, line in enumerate(infile):
	l = line.split()
	written = 0
	res = ''.join(i for i in l[0] if not i.isdigit())
	if resid == res:
		resno = l[0].replace(resid, "")
		if resno not in old_l[0]:
			start_idx = idx-1
			no = 0
			old_resno = old_l[0].replace(resid, "")
		if l[1] == atom_seq[no]:
			if l[2] != start_idx+no:
				line = str_replace(line, 15, 20, add_spaces(start_idx+no, 5))
			outfile.write(line)
			written = 1
			no += 1
		else:
			saved_lines.append(line)
			written = 1
	s = 0
	while 1:
		if s < len(saved_lines):
			#print(saved_lines[s].split()[1], " ", atom_seq[no])
			if saved_lines[s].split()[1] == atom_seq[no]:
				if l[2] != start_idx+no:
					sline = str_replace(saved_lines[s], 15, 20, add_spaces(start_idx+no, 5))
				outfile.write(sline)
				del saved_lines[s]
				s = 0
				no += 1
			else:
				s += 1
		else:
			break
	old_l = line.split()
	if resid != l[0] and written == 0 :
		outfile.write(line)
   
infile.close()
outfile.close()
sys.exit(f"Output succesfully written to the {conv_pdb} file.")


