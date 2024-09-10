# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:50:37 2021

@author: Forty
"""

"""
import sys
if __name__ == "__main__":
    if len(sys.argv)!=3:
        sys.exit("Wrong number of arguments given. Correct syntax: paths_intersection.py path1 path2")
    else:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
""";

#input (info about protein chains and numbers of starting/ending amino acids):
chains_list = ["A", "B"]
protein_boundary_nums = [[37,373], [37, 373]]
last_end = 0
borders = []
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
	
for n in range(5):
	path1 = f"./{n}/fit_paths/A_L185-A_H91.out.out"
	path2 = f"./{n}/fit_paths/B_D264-A_H91.out.out"

	name1 = path1.split("/")[-1].rstrip('.out')
	name2 = path2.split("/")[-1].rstrip('.out')

	file1 = open(path1, "r")
	lines1 = file1.readlines()
	file1.close()

	file2 = open(path2, "r")
	lines2 = file2.readlines()
	file2.close()

	nodes1_set = set(map(int, lines1[6].split(',')[:-1]))
	nodes2_set = set(map(int, lines2[6].split(',')[:-1]))
	nodes_dict = {}
				
	def write_dict(set1 : set):
		for v in chains_list:
			nodes_dict[v] = []
		for i, number in enumerate(set1):
			vmd_input = nodenum_to_vmd(number)
			nodes_dict[vmd_input[0]].append(vmd_input[1])
		for chain in chains_list:
			if nodes_dict[chain]:
				outfile.write(f"(chain {chain} and (resid ")
				for i in nodes_dict[chain][:-1]:
					outfile.write(f"{i} or resid ")
				if chain != chains_list[-1] and nodes_dict[chains_list[-1]]:
					outfile.write(f"{nodes_dict[chain][-1]})) or ")
				else:
					outfile.write(f"{nodes_dict[chain][-1]}))\n")

	out = path1.removesuffix(name1+'.out.out')+f"intsec_opt_{name1}_{name2}.txt"
	outfile = open(out, "w")
	#sorted(list(nodes2_set))
	outfile.write(f"Nodes for path {name1}:\n")
	write_dict(nodes1_set)
	outfile.write(f"Nodes for path {name2}:\n")
	write_dict(nodes2_set)
	nodes_intersection = nodes1_set.intersection(nodes2_set)
	outfile.write(f"Intersecting nodes in {name1} and {name2}:\n")	
	write_dict(nodes_intersection)	        
	outfile.close()
	print(f"Output file written under the path {out}")

