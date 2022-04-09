# wersja 2- zmioniony python na python3 

# pliki inputowe to sesja w pymolu i plik tekstowy z parami atomow
# aby stworzyc sesje w pymolu zawierajaca obiekt o wielu states - join_states nazwa_output, all lub zamiast all (..., ...) lub konkretna nazwa

#!/usr/bin/env python3
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import sys, os
import pymol

directory = sys.argv[1]
dir_path = os.path.abspath(directory)
ref_structure = sys.argv[2]


pymol.finish_launching()

files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)

files.remove(ref_structure)
files.sort()

f = os.path.abspath(ref_structure)
pymol.cmd.load(f, ref_structure)


for j in files:
    f = os.path.abspath(j)
    j=j[:-4]
    pymol.cmd.load(f, j)

output_dist=open('output_rmsd.dat','w')
output_dist.write(dir_path)
output_dist.write("\n\n") 

for s in range(len(files)):
    rmsd = []
    protein = f"model_{s+1}_0001"
    trna = f"lig_{s+1}.pdb"
    pymol.cmd.alter(f"/{protein}//B" , "chain='E'")
    pymol.cmd.create(f"merged_mol", f"{protein} | {trna}") 
    pymol.cmd.delete (f"name = {protein}" )
    pymol.cmd.delete (f"name = {trna}" )
    #align
    align_whole=pymol.cmd.align(f'{ref_structure}','merged_mol')
    rmsd.append(list(align_whole))
    align_ca_p=pymol.cmd.align(f'{ref_structure} and name CA+P','merged_mol and name CA+P')
    rmsd.append(list(align_ca_p))
    #super
    super_whole=pymol.cmd.super(f'{ref_structure}','merged_mol')
    rmsd.append(list(super_whole))
    super_ca_p=pymol.cmd.super(f'{ref_structure} and name CA+P','merged_mol and name CA+P')
    rmsd.append(list(super_ca_p))     
    rmsd_str = "\n".join("\t".join(str(i) for i in j) for j in rmsd) 
    pymol.cmd.delete ("name = 'merged_mol'" )
    output_dist.write(f'{files[s]}\n{rmsd_str}\n')
    #output_dist.write('-'*50)

output_dist.close()


