#!/usr/bin/env python3
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import sys, os
import pymol

if __name__ == "__main__":
    if len(sys.argv) == 4:
        directory = sys.argv[1]
        dir_path = os.path.abspath(directory)
        ref_structure = sys.argv[2]
        bonds = sys.argv[3]
    else:
        sys.exit(
            "Wrong number of arguments given. Correct syntax: dist_res_verMF.py directory ref_structure bonds.txt")

pymol.finish_launching()

files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)

files.remove(ref_structure)
files.sort()

f = os.path.abspath(ref_structure)
pymol.cmd.load(f, ref_structure)

dane_plik = open(bonds,'r')
lines = dane_plik.readlines()
dane_plik.close()
lines.remove(lines[0])

lines = [i[:-1] for i in lines]
dane_lista = []

for l in lines:
    dane_lista.append(l.split('\t'))
    # l.split wyrzuca liste stringow z linii l, a nastepnie append dodaje ta liste stringow do dane_lista

for j in files:
    f = os.path.abspath(j)
    j = j[:-4]
    pymol.cmd.load(f, j)

output_dist=open('output_dist.dat','w')
output_dist.write(dir_path)
output_dist.write("\n\n") 

for s in files:
    rmsd = []
    name = s.split("/")[-1].rstrip(".pdb")
    number = name.split("_")[-1]
    protein = f"model_{number}_0001"
    trna = f"lig_{number}.pdb"
    pymol.cmd.alter(f"/{protein}//B" , "chain='E'")
    pymol.cmd.create(f"merged_mol", f"{protein} | {trna}") 
    pymol.cmd.delete (f"name = {protein}" )
    pymol.cmd.delete (f"name = {trna}" )
    #RMSD
    align_whole=pymol.cmd.align(f'{ref_structure}','merged_mol')
    rmsd.append(list(align_whole)[3])
    rmsd_str = (" ".join(str(x) for x in rmsd))
    output_dist.write(f'START\t{name}\t{rmsd_str}\n')

    for i in range(len(dane_lista)):
        bond=dane_lista[i]
        distance=pymol.cmd.get_distance(atom1=f'/merged_mol//{bond[0]}/{bond[1]}/{bond[2]}', atom2=f'/merged_mol//{bond[3]}/{bond[4]}/{bond[5]}')
        
        output_dist.write("\t".join((lines[i], str(distance))))
        output_dist.write('\n')
    pymol.cmd.delete ("name = 'merged_mol'" )
    output_dist.write(f'END\t{name}\n\n')
    #output_dist.write('-'*50)

output_dist.close()


