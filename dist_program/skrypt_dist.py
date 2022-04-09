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
if len(ref_structure) != 4:
    raise ValueError("ref_structure must be a pdb code")
bonds=sys.argv[3]


pymol.finish_launching()

output_dist=open('output_dist.dat','w') #pliki tekstowe w ktorych bede zapisywala odleglosci
output_dist.write(dir_path)
output_dist.write("\n\n") 
output_dist.write('file, distance')
output_dist.write("\n\n") 

files = []
for file in os.listdir(directory):
    if file.endswith(".pdb"):
        files.append(file)
        continue
    else:
        continue

print (files)

for j in range(len(files)):
    file = files[j]
    f = os.path.abspath(file)
    
    pymol.cmd.load(f, file)
    pymol.cmd.create('mol',file,0,-1)

pymol.cmd.fetch(ref_structure, 'ref') 

states = pymol.cmd.count_states('all') 

#dane_plik - informacja jakie wiazania bedziemy liczyc - w formie pliku txt skopiowanego z excela

dane_plik=open(bonds,'r')
lines=dane_plik.readlines()
lines.remove(lines[0])

for i in range(len(lines)):
    lines[i]=lines[i][:-1]

dane_lista = []

for l in lines:
    dane_lista.append(l.split('\t')) #l.split wyrzuca liste stringow z linii l, a nastepnie append dodaje ta liste stringow do dane_lista


for s in range(1,states+1):
    file =files[s-1] 
    protein = [file]

    #RMSD
    cealn_whole=pymol.cmd.cealign('ref and name CA+P','mol and name CA+P', mobile_state=s)
    protein.append(cealn_whole['RMSD'])
    output_dist.write(" ".join(str(x) for x in protein))
    output_dist.write("\n")  

    for i in range(len(dane_lista)):
        bond=dane_lista[i]
        distance=pymol.cmd.get_distance(atom1=f'/mol//{bond[0]}/{bond[1]}/{bond[2]}', atom2=f'/mol//{bond[3]}/{bond[4]}/{bond[5]}', state=s)
        
        output_dist.write("\t\t".join((lines[i], str(distance))))
        output_dist.write('\n')
    
    output_dist.write('\n')
    output_dist.write('-'*50)
    output_dist.write('\n')    

#output_dist.write(" ".join(str(x) for x in protein))
#output_dist.write("\n")
#
output_dist.close()


