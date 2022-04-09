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
bonds=sys.argv[3]


pymol.finish_launching()

output_dist=open('output_dist.dat','w') #pliki tekstowe w ktorych bede zapisywala odleglosci
output_dist.write(dir_path)
output_dist.write("\n\n") 

files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)

files.remove(ref_structure)
files.sort()

f = os.path.abspath(ref_structure)
pymol.cmd.load(f, ref_structure)

dane_plik=open(bonds,'r')
lines=dane_plik.readlines()
dane_plik.close()
lines.remove(lines[0])

lines = [i[:-1] for i in lines]

dane_lista = []

for l in lines:
    dane_lista.append(l.split('\t')) #l.split wyrzuca liste stringow z linii l, a nastepnie append dodaje ta liste stringow do dane_lista

for j in files:
    f = os.path.abspath(j)
    j=j[:-4]
    pymol.cmd.load(f, j)

rmsd = []
for s in range(len(files)):
    protein = f"model_{s+1}_0001"
    trna = f"lig_{s+1}"
    pymol.cmd.alter(f"/{protein}//B" , "chain='E'")
    pymol.cmd.create(f"merged_mol", f"{protein} | {trna}") 

    #RMSD
    cealn_whole=pymol.cmd.cealign(f'{ref_structure} and name CA+P','merged_mol and name CA+P')
    #cealn_whole=pymol.cmd.cealign('ref_structure','mol', mobile_state=s)
    rmsd.append(cealn_whole['RMSD'])
    rmsd = (" ".join(str(x) for x in rmsd))
    output_dist.write(f'START\t{files[s]}\t{rmsd}\n')

    for i in range(len(dane_lista)):
        bond=dane_lista[i]
        distance=pymol.cmd.get_distance(atom1=f'/merged_mol//{bond[0]}/{bond[1]}/{bond[2]}', atom2=f'/merged_mol//{bond[3]}/{bond[4]}/{bond[5]}')
        
        output_dist.write("\t".join((lines[i], str(distance))))
        output_dist.write('\n')
    
    output_dist.write(f'END\t{files[s]}\n\n')
    #output_dist.write('-'*50)

output_dist.close()


