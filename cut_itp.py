#!/usr/bin/env python3
"""
Created on 25.07.2022
@author: Mateusz Fortunka
"""

import sys

if __name__ == "__main__":
	if len(sys.argv)!=2:
		sys.exit("Wrong number of arguments given. Correct syntax: cut_itp.py file.itp")
	else:
		file_itp=sys.argv[1]
		if file_itp[-4:] != '.itp':
			print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: cut_itp.py file.itp")

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
#file_itp = "/home/matifortunka/Documents/ACC/ACC_md/PAA/aa_0.itp"

infile = open(file_itp, "r")
lines = infile.readlines()
infile.close()

res = {}

for i, v in enumerate(lines):
    if "[ atoms ]" in v:
        no = 0
        for j, l in enumerate(lines[i+3:]):
            s = l.split()
            if len(s) == 0:
                break
            no+=1
            res[s[0]] = no 
            lines[j+i+3] = str_replace(lines[j+i+3], 3, 6, add_spaces(res[s[0]],3))
            lines[j+i+3] = str_replace(lines[j+i+3], 42, 45, add_spaces(res[s[0]],3))
        continue
    elif "[ bonds ]" in v or "[ pairs ]" in v :
        j=0
        while 1:
            l = lines[i+2+j]
            s = l.split()
            if len(s) == 0:
                break
            elif s[0] not in res.keys() or s[1] not in res.keys():
                del lines[j+i+2]
            else:
                lines[j+i+2] = str_replace(lines[j+i+2], 2, 5, add_spaces(res[s[0]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 8, 11, add_spaces(res[s[1]],3))
                j+=1
        continue
    elif "[ angles ]" in v:
        j=0
        while 1:
            l = lines[i+2+j]
            s = l.split()
            if len(s) == 0:
                break
            elif s[0] not in res.keys() or s[1] not in res.keys() or s[2] not in res.keys():
                del lines[j+i+2]
            else:
                lines[j+i+2] = str_replace(lines[j+i+2], 2, 5, add_spaces(res[s[0]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 8, 11, add_spaces(res[s[1]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 14, 17, add_spaces(res[s[2]],3))
                j+=1
        continue
    elif "[ dihedrals ]" in v:
        j=0
        while 1:
            l = lines[i+2+j]
            s = l.split()
            if len(s) == 0:
                break
            elif s[0] not in res.keys() or s[1] not in res.keys() or s[2] not in res.keys() or s[3] not in res.keys():
                del lines[j+i+2]
            else:
                lines[j+i+2] = str_replace(lines[j+i+2], 2, 5, add_spaces(res[s[0]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 8, 11, add_spaces(res[s[1]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 14, 17, add_spaces(res[s[2]],3))
                lines[j+i+2] = str_replace(lines[j+i+2], 20, 23, add_spaces(res[s[3]],3))
                j+=1
        continue

conv = file_itp[0:-4]+"_n"+".itp"
outfile = open(conv, "w")
for line in lines:
    outfile.write(line)

outfile.close()
sys.exit(f"Output succesfully written to the {conv} file.")


