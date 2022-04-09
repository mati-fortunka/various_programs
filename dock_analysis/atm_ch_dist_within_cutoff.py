#!/usr/bin/env python3
"""
@author: Mateusz Fortunka

Program calculates distances between every pair of atoms read from pdb file within cutoff. There is option to ommit all atoms of one type.
"""
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import sys, os
import pymol

if __name__ == "__main__":
    if len(sys.argv) == 4:
        structure = sys.argv[1]
        cutoff = float(sys.argv[2])
        ommit_atm = sys.argv[3]
    elif len(sys.argv) == 3:
        structure = sys.argv[1]
        cutoff = float(sys.argv[2])
        ommit_atm = None
    else:
        sys.exit("Wrong number of arguments given. Correct syntax: atm_ch_dist_within_cutoff.py file.pdb cutoff (atom_to_ommit)")

if structure[-4:] != '.pdb':
    print("Input file has a wrong extension. Nevertheless, the program will try to open it. However, the results can be incorrect! Correct syntax: pdb_translate file.pdb promals_hits")

pymol.finish_launching()

out_human = f"dist_{structure[:-4]}_{cutoff}.dat"
out_indist =  f"indist_{structure[:-4]}_{cutoff}.dat"
output_human = open(out_human,'w')
output_human.write('no1\tname1\tres1\tch1\tresno1\tvs\tno2\tname2\tres2\tch2\tresno2\tdist\n')
output_indist = open(out_indist ,'w')
output_indist.write('chain1\tnres1\tat1\tchain2\tnres2\tat2\n')


f = os.path.abspath(structure)
pymol.cmd.load(f, structure)

atomy_pdb=open(structure,'r')
lines=atomy_pdb.readlines()
atomy_pdb.close()

lines = [i[:-1] for i in lines]
atomy_lista = []


for l in lines:
    if l[:4] == "ATOM" or l[:6] == "HETATM":
        if ommit_atm is None or l.split()[2][0] != ommit_atm:
            atomy_lista.append(l.split())

old_res=""
flag=0
for a1 in atomy_lista:
    atomy_lista.remove(a1)
    for a2 in atomy_lista:
        if a1[4] == a2[4]: continue
        res = "".join([a2[4], a2[5]])
        if flag == 1 and res == old_res:
            old_res = res
            continue
        flag = 0
        old_res = res
        distance=pymol.cmd.get_distance(atom1=f'/{structure}//{a1[4]}/{a1[5]}/{a1[2]}', atom2=f'/{structure}//{a2[4]}/{a2[5]}/{a2[2]}')
        if distance > cutoff+10:
            flag = 1
        elif distance <= cutoff:
            output_human.write("\t".join(["\t".join(a1[1:6]),"vs", "\t".join(a2[1:6]), str(distance)]))
            output_human.write('\n')
            output_indist.write("\t".join([a1[4], a1[5], a1[2], a2[4], a2[5], a2[2]]))
            output_indist.write('\n')


output_human.close()
output_indist.close()

