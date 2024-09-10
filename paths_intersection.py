# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:50:37 2021

@author: Forty
"""

import sys

if __name__ == "__main__":
    if len(sys.argv)!=3:
        sys.exit("Wrong number of arguments given. Correct syntax: paths_intersection.py path1 path2")
    else:
        path1 = sys.argv[1]
        path2 = sys.argv[2]

#input (info about protein chains and numbers of starting/ending amino acids):
chains_list = ["A", "B"]
protein_boundary_nums = [[37,373], [37, 373]]

#path1 = "/home/matifortunka/Documents/JS/publication_YARS/apoenzyme/2pid_mut/vmd_ss/2/fit_paths/A_L185-A_H91.out.out"
#path2 = "/home/matifortunka/Documents/JS/publication_YARS/apoenzyme/2pid_mut/vmd_ss/2/fit_paths/B_D264-A_H91.out.out"

name1 = path1.split("/")[-1].rstrip('.out')
name2 = path2.split("/")[-1].rstrip('.out')
borders = []
last_end = 0
for i in protein_boundary_nums:
    start = i[0]
    borders.append(i[1] - start + last_end)
    last_end = borders[-1] + 1
    
def nodenum_to_vmd(node_num):
    for border_num, border in enumerate(borders):
        if node_num <= border:
            if border_num > 0:
                node_num-=(borders[border_num-1]+1)
            output = [chains_list[border_num], node_num+protein_boundary_nums[border_num][0]]
            return output
    return "Error! Node number is out of the given protein boundaries."
        

file1 = open(path1, "r")
lines1 = file1.readlines()
file1.close()

file2 = open(path2, "r")
lines2 = file2.readlines()
file2.close()

nodes1_set = set(map(int, lines1[6].split(',')[:-1]))
nodes2_set = set(map(int, lines2[6].split(',')[:-1]))

for l in lines1[6:]:
    if l[0:6]!="Number":
        nodes = set(map(int, l.split(',')[:-1]))
        nodes1_set.update(nodes)
        
for l in lines2[6:]:
    if l[0:6]!="Number":
        nodes = set(map(int, l.split(',')[:-1]))
        nodes2_set.update(nodes)

out = path1.removesuffix(name1+'.out.out')+f"intersection_{name1}_{name2}.txt"
outfile = open(out, "w")
#sorted(list(nodes2_set))
outfile.write(f"Nodes for path {name1}:\n")
for i, number in enumerate(nodes1_set):
    vmd_input = nodenum_to_vmd(number)
    if i < len(nodes1_set)-1:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]}) or ")
    else:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]})\n")
            
outfile.write(f"Nodes for path {name2}:\n")
for i, number in enumerate(nodes2_set):
    vmd_input = nodenum_to_vmd(number)
    if i < len(nodes2_set)-1:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]}) or ")
    else:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]})\n")

nodes_intersection = nodes1_set.intersection(nodes2_set)
outfile.write(f"Intersecting nodes in {name1} and {name2}:\n")
for i, number in enumerate(nodes_intersection):
    vmd_input = nodenum_to_vmd(number)
    if i < len(nodes_intersection)-1:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]}) or ")
    else:
        outfile.write(f"(chain {vmd_input[0]} and resid {vmd_input[1]})\n")

outfile.close()
print(f"Output file written under the path {out}")

