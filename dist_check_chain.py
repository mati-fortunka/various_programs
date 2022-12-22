#!/usr/bin/env python3
import __main__

__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI
import sys, os
import pymol
from Contacts import Contacts

if __name__ == "__main__":
    if len(sys.argv) == 4:
        directory = sys.argv[1]
        dir_path = os.path.abspath(directory)
        bonds = sys.argv[2]
        no_con_tocheck = int(sys.argv[3])
    else:
        sys.exit(
            "Wrong number of arguments given. Correct syntax: dist_check_chain.py directory contacts_2vers.txt "
            "no_of_contacts_to_distinguish")

pymol.finish_launching()


def find_many_lno_str(s: str, lines: list, how_many: int = 1):
    lines_nums = []
    while len(lines_nums) < how_many:
        for i, v in enumerate(lines):
            if s in v:
                lines_nums.append(i)
    return lines_nums


def dist(contact: list):
    return pymol.cmd.get_distance(atom1=f'/merged_mol//{contact[0]}/{contact[1]}/{contact[2]}',
                                  atom2=f'/merged_mol//{contact[3]}/{contact[4]}/{contact[5]}')


files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)
files.sort()

con_file = open(bonds, 'r')
contacts = con_file.readlines()
con_file.close()

vers_no = find_many_lno_str("#version", contacts, 2)
vers_val = [contacts[v].split()[1] for v in vers_no]
con = [Contacts(contacts[vers_no[0] + 2:vers_no[1] - 1]), Contacts(contacts[vers_no[1] + 2:-1])]

for j in files:
    f = os.path.abspath(j)
    j = j[:-4]
    pymol.cmd.load(f, j)

output_dist = open('output_dist.dat', 'w')
output_dist.write(dir_path)
output_dist.write("\n\n")

for s in files:
    name = s.split("/")[-1].rstrip(".pdb")
    number = name.split("_")[-1]
    protein = f"model_{number}_0001"
    trna = f"lig_{number}.pdb"
    # pymol.cmd.alter(f"/{protein}//B", "chain='E'")
    pymol.cmd.create(f"merged_mol", f"{protein} | {trna}")
    pymol.cmd.delete(f"name = {protein}")
    pymol.cmd.delete(f"name = {trna}")

    mean_1 = 0
    mean_2 = 0
    for i in range(no_con_tocheck):
        mean_1 += dist(con[0][i])
        mean_2 += dist(con[1][i])
    mean_1 = mean_1 / no_con_tocheck
    mean_2 = mean_2 / no_con_tocheck
    if mean_2 > mean_1:
        ver = 0
    else:
        ver = 1

    output_dist.write(f'START\t{name}\t{vers_val[ver]}\n')

    for c in con[ver]:
        c[-1] = str(dist(c))
        output_dist.write("\t".join(c))
        output_dist.write('\n')
    pymol.cmd.delete("name = 'merged_mol'")
    output_dist.write(f'END\t{name}\n\n')
output_dist.close()
