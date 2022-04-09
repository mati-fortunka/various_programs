#!/usr/bin/env python3
"""
Created on 04.11.2021
@author: Mateusz Fortunka

Python version >= 3.6 needed because of fstring usage.


This program takes output from the script "promals_hits.py" (which uses Promals3D alignment output of two proteins) and a list of aminoacids taken from the second polypeptide (order is important, especially in Promals output). Then compares them and gives a list of aminoacids from the first protein that were homologic to the aminoacids from the second protein and present in the given list. In short it translates specified aminoacids from one protein (usually homologic one) to another (target protein). Obviously not all aminoacids from the list are translated, because sometimes there is no alignment (e.g. aa is aligned to a gap when one protein is longer or just do not match the other aminoacid).

The script was meant to prepare input file with specified residues to an advanced docking. For example it is useful when studied crystal structure does not include RNA or ligand, but you can find homologic structure with both - protein and RNA/ligand. Using promals_translate.py you can compare a target protein with the homologic one and e.g. get contacts in the first complex. It improves or even enables proper docking at all.


Usage:
./promals_translate.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt docking_server_name (--max_cfac default_confidence_factor) (--max_cfac max_confidence_factor) (--s_flag strict_flag)


Example usages:
python3.8 promals_translate.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt MPRDOCK
python3.7 promals_translate.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt 3dRPC --c_fac 0.2 --max_cfac 0.6
python3.7 promals_translate.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt 3dRPC --c_fac 0.5 --max_cfac 1 --s_flag 0


Explanation:
 - promals_output_target_vs_homologic.txt: Output from "promals_hits.py" (Promals3D output visualised in one line -> copied to the .txt file -> used as an imput to promals_hits.py"). Four columns are necessary: target_prot_no, target_prot_aa, homologic_prot_no and consensus_symbol.
 - aas_list_from_homologic.txt: List of aminoacids from homologic protein e.g. which are in contact with docked RNA/ligand. Two columns are necessary: "res_no" and "chain". It can be prepared with "list_res+aa_code_conv.py" program.
 - docking_server_name: Name of server which you want use for docking. Three recomended for protein-tRNA docking are: MPRDock, 3dRPC and HDock (similar to MPRDock). For protein-protein: HDock, PatchDock. P3Dock do not accept specified residues for docking. NPDock still not working. Thus the last two are not in options of the output format. Additional option - "comma" to get list of translated aminoacids separated with commas and spaces. (Some shortcuts of server names are allowed e.g. 3d, h, mpr, patch, p, m).
 - c_fac (default confidence factor): Essential on 3dRPC server. By default it is assigned to every aligned aminoacid but not fully (e.g. D -> E - negatively charged, D -> K - charged,  R -> T - polar, I -> V - aliphatic)
 - max_cfac (maximal confidence factor): Essential on 3dRPC server. By default it is assigned to every fully aligned aminoacid (e.g. L -> L, G -> G, K -> K).
- s_flag (strict flag): If equal to 0 program takes all aminoacids from homologic protein specified in the input file "aas_list_from_homologic.txt" and writes translated ones with all chains present in the target protein. Else program writes only aminoacids in with chain from homologic polypeptide specified in "aas_list_from_homologic.txt" file. This option 0 is useful only for proteins with multiple alike chains (dimers, trimers, tetramers etc.). Use this option carefully - names of chains must be the same in both protein files serving as docking input!

Keep in mind that format for HDock and MPRDock is the same. There is also additional option (comma/c/,) of output with commas and spaces between aminoacids, which can be used on both mentioned servers pasting saved text in file to the right box on site instead of uploading a file.

You can manually change values assigned to every type of Promals3D hit in code (dictionary "consensus_dict"). By default full alignments have "max_confidence_factor" and the rest have the same "default_confidence_factor". These both values can be specified when running the program. They are essential on the 3dRPC server, but can be also specified on the others. Hovever, optional addition of confidence factors in HDOCK, MPRDock, PatchDock and comma separated output is not yet implemented (it was not neccessary so far).

"""

import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 10:
        sys.exit("Wrong number of arguments given. Correct syntax:"
                 "promals_translate.py promals_output_target_vs_homologic.txt aas_list_from_homologic.txt"
                 " docking_server_name (--c_fac default_confidence_factor) (--max_cfac max_confidence_factor) (--s_flag strict_flag)")
    else:
        hits = sys.argv[1]
        aa_list = sys.argv[2]
        server = sys.argv[3]
        strict_flag = ""
        c_fac = "0.5"
        max_cfac = "1"
        for argc in range(4, len(sys.argv), 2):
                if sys.argv[argc] not in ["--c_fac", "--max_cfac", "--s_flag"]:
                        sys.exit(f"Unknown option {sys.argv[argc]}. Check the spelling.")
                elif sys.argv[argc] == "--c_fac":
                        c_fac = sys.argv[argc+1]
                elif sys.argv[argc] == "--max_cfac":
                        max_cfac = sys.argv[argc+1]
                elif sys.argv[argc] == "--s_flag":
                        strict_flag = sys.argv[argc+1]
		

def consensus_to_conf(ch):
    consensus_dict = {"l": c_fac, "@": c_fac, "h": c_fac, "o":c_fac,
                      "p":c_fac, "t":c_fac, "s":c_fac, "b":c_fac,
                      "+":c_fac, "-":c_fac, "c":c_fac}
    if ch.isupper():
        return max_cfac
    else:
        return consensus_dict[ch]

hits_tab = pd.read_csv(hits, sep="\t")
aa_tab = pd.read_csv(aa_list, sep="\t")
aa_tab["res_no"] = aa_tab["res_no"].astype(int)
aa_tab = aa_tab.sort_values(["res_no", "chain"])
out = "translated_"+server+strict_flag+".txt"
#out = hits[0:-4]+"_"+aa_list[0:-4]+server+strict_flag+".txt"

output_list = []
aa1_list = []
confidence = []

if strict_flag == str(0):
    chains = aa_tab['chain']
    chains = chains.unique()
    old_v = -1
    for i, v in enumerate(hits_tab["homologic_prot_no"].tolist()):
        if v == old_v: continue
        if v in aa_tab['res_no'].tolist():
            for c in chains:
                output_list.append([str(hits_tab["target_prot_no"][i]), c])
                aa1_list.append(hits_tab["target_prot_aa"][i])
                confidence.append(hits_tab["consensus_symbol"][i])
            old_i = i
else:
    for i, v in enumerate(hits_tab["homologic_prot_no"].tolist()):
        aa_hit = aa_tab[(aa_tab['res_no'] == v)]
        for it in aa_hit["chain"].tolist():
            output_list.append( [str(hits_tab["target_prot_no"][i]), it])
            aa1_list.append(hits_tab["target_prot_aa"][i])
            confidence.append(hits_tab["consensus_symbol"][i])

outfile = open(out, "w")

if server in ["HDOCK", "hdock", "h", "H","MPRDOCK", "mprdock", "mpr", "m", "M"]:
    for i in output_list:
        line = ":".join(i)
        outfile.write(line + '\n')

elif server in ["3dRPC", "3drpc", "3d", "rpc", "3DRPC", "3Drpc", "3"]:
    c_list = [consensus_to_conf(i) for i in confidence]
    for j, v in enumerate(output_list):
        line = "".join(v[::-1])
        line += aa1_list[j] + " " + str(c_list[j])
        outfile.write(line + '\n')

elif server in ["PATCHDOCK", "patchdock", "patch", "PatchDock", "PatchDOCK", "p", "P"]:
    for i in output_list:
        line = " ".join(i)
        outfile.write(line + '\n')

elif server in ["comma", "c", ","]:
    for i in output_list[:-1]:
        pair = ":".join(i)
        outfile.write(pair + ', ')
    outfile.write(":".join(output_list[-1]))

outfile.close()
sys.exit(f"Output succesfully written to {out} file.")

