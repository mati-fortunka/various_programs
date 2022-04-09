# Initialize
load yourProtein
cmd.wizard("mutagenesis")
cmd.do("refresh_wizard")

# To get an overview over the wizard API:
for i in dir(cmd.get_wizard(): print i

# lets mutate residue 104 to GLN
cmd.get_wizard().set_mode("GLN")
cmd.get_wizard().do_select("104/")

# Select the rotamer
cmd.frame(11)

# Apply the mutation
cmd.get_wizard().apply()
