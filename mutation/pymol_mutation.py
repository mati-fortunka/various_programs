#!/usr/bin/env python3
# Modified by Mateusz Fortunka
# 14.03.2022
# program was designed to mutate aminoacids, but also works with nucleotides

import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import sys, os
from pymol import cmd

if __name__ == "__main__":
	if len(sys.argv) == 3:
		prot_file = sys.argv[1]
		mut_file = sys.argv[2]
	else:
		sys.exit("Wrong number of arguments given. Correct syntax: pymol_mutation.py protein_file file_with_mutations")

def mutate(sele):
	cmd.get_wizard().do_select(f"{sele}") #my example: "A002/A/15/" if more than one protein, its name is required
	cmd.frame(1)	#choose 1st rotamer (the rotamers are ordered according to their frequencies of occurrence in proteins)
	cmd.get_wizard().apply()

f = os.path.abspath(prot_file)
cmd.load(f, prot_file)

with open(mut_file, "r") as f:
	mut_table = f.readlines()
	
mut_table.remove(mut_table[0])
mut_table = [i[:-1].split() for i in mut_table]

for mut in mut_table:

	# mutagenesis mode
	cmd.wizard("mutagenesis")
	cmd.do("refresh_wizard")

	selection, aminoacid = mut
	cmd.get_wizard().set_mode(aminoacid)	# set aminoacid which wizard is mutating residues to
	resid = selection.split("/")[-2]

	# Mutate
	# if given a range of residues we need loop
	if "-" in resid:
		start, end = resid.split("-")
		for j in range(int(start), int(end)+1):
			new_selection = selection.replace(f"{resid}/", "") + str(j) + "/"
			mutate(new_selection)
	else:
		mutate(selection) 

# save mutated protein
cmd.save("mutated.pdb", prot_file.replace(".pdb", ""))

   
