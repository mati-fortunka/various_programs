#
# Rozaimi Razali <rozaimirazali@gmail.com>
#
# Modified by PhD. Student Mario Sergio Valdes Tresanco
# June 26 2018


prot_list = [['1shr', 'B']] # example [['1nzp', 'A'], ['4oxn', 'B']] # how many proteins do you want
residues_list = [36, 37, 38] # example [3, 35, 100] # only 3 residues
aminoacid = 'ALA' # only one


for prot in prot_list:
    #Initialize
    # fetch protein by PDB code
    cmd.fetch(prot[0], async=0)
    # mutagenesis mode
    cmd.wizard("mutagenesis")
    cmd.do("refresh_wizard")
    
    # create 7 object 
    for obj in range(1,8):
        cmd.create('obj_%s' % obj, prot[0])
    # Mutate
    cmd.get_wizard().set_mode(aminoacid)
    
    #for obj_1 (mutation in all residues)   
    cmd.get_wizard().do_select("/obj_1//%s/%d" % (prot[1], residues_list[0]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.get_wizard().do_select("/obj_1//%s/%d" % (prot[1], residues_list[1]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.get_wizard().do_select("/obj_1//%s/%d" % (prot[1], residues_list[2]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[0], residues_list[1], residues_list[2], aminoacid),'obj_1')
    
    #for obj_2 (mutation in residue # 1 and 2)
    cmd.get_wizard().do_select("/obj_2//%s/%d" % (prot[1], residues_list[0]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.get_wizard().do_select("/obj_2//%s/%d" % (prot[1], residues_list[1]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[0], residues_list[2], aminoacid),'obj_2')
    
    #for obj_3 (mutation in residue # 1 and 3)
    cmd.get_wizard().do_select("/obj_3//%s/%d" % (prot[1], residues_list[0]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.get_wizard().do_select("/obj_3//%s/%d" % (prot[1], residues_list[2]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[0], residues_list[2], aminoacid),'obj_3')
    
    #for obj_4 (mutation in residue # 2 and 3)
    cmd.get_wizard().do_select("/obj_4//%s/%d" % (prot[1], residues_list[1]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.get_wizard().do_select("/obj_4//%s/%d" % (prot[1], residues_list[2]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[1], residues_list[2], aminoacid),'obj_4')
    
    #for obj_5 (mutation in residue # 1)
    cmd.get_wizard().do_select("/obj_5//%s/%d" % (prot[1], residues_list[0]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[0], aminoacid),'obj_5')
    #for obj_6 (mutation in residue # 2)
    cmd.get_wizard().do_select("/obj_6//%s/%d" % (prot[1], residues_list[1]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[1], aminoacid),'obj_6')
    #for obj_7 (mutation in residue # 3)
    cmd.get_wizard().do_select("/obj_7//%s/%d" % (prot[1], residues_list[2]))
    cmd.frame(1)
    cmd.get_wizard().apply()
    cmd.save("%s_%s_%s_x_%s.pdb" % (prot[0], prot[1], residues_list[2], aminoacid),'obj_7')

    # Done
    cmd.set_wizard()	#refreshes wizard
    if len(prot_list) == 1:
        continue
    else:
        cmd.reinitialize()  #reinitializes/cleans pymol
   
