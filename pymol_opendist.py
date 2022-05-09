#!/usr/bin/env python3
#import __main__
#__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import sys, os
import pymol
import pandas
from Contacts import Contacts

if __name__ == "__main__":
    if len(sys.argv) == 3:
        pdb_file = sys.argv[1]
        con_file = sys.argv[2]
    else:
        sys.exit(
            "Wrong number of arguments given. Correct syntax: pymol_opendist.py pdb_file contacts_file")

#pymol.finish_launching()

con = open(con_file, "r")
contacts = Contacts(con.readlines()[1:])
con.close()

fpath = os.path.abspath(pdb_file)
name = fpath.split("/")[-1]
name = name[:-4]
pymol.cmd.load(fpath, name)

s = name.split("_")[-1]

protein = f"model_{s}_0001"
trna = f"lig_{s}.pdb"
pymol.cmd.alter(f"/{protein}//B" , "chain='E'")
pymol.cmd.create(f"merged_mol", f"{protein} | {trna}")
pymol.cmd.delete (f"name = {protein}" )
pymol.cmd.delete (f"name = {trna}" )

for i, v in enumerate(contacts):
    distance=pymol.cmd.distance(f"dist_{i}" , f'/merged_mol//{v[0]}/{v[1]}/{v[2]}', f'/merged_mol//{v[3]}/{v[4]}/{v[5]}')

pymol.cmd.save(f"{protein[:-5]}.pse")

