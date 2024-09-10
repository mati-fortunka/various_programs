import __main__
import pymol
__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI
from pymol import cmd

# Initialize PyMOL
pymol.finish_launching()

def find_gaps(pdb_file):
    # Load the PDB file
    cmd.load(pdb_file, "protein1")

    # Get all chains in the structure
    chains = cmd.get_chains("protein1")

    gaps = []

    # Iterate through each chain
    for chain in chains:
        # Get all residues in the chain
        model = cmd.get_model(f"protein1 and chain {chain}")
        residue_ids = set()

        for atom in model.atom:
            resi = atom.resi
            residue_ids.add(int(resi))

        # Convert to sorted list
        residue_ids = sorted(residue_ids)

        # Find gaps
        for i in range(len(residue_ids) - 1):
            current_residue = residue_ids[i]
            next_residue = residue_ids[i + 1]
            if next_residue != current_residue + 1:
                gaps.append((current_residue + 1, next_residue - 1, chain))

    return gaps

def fill_gaps(gaps, pdb_file2):
    # Load the second PDB file
    cmd.load(pdb_file2, "protein2")
    
    # Align the second protein to the first
    cmd.align("protein2", "protein1")
    
    filled_residues = []

    for gap in gaps:
        start_residue = gap[0]
        end_residue = gap[1]
        chain = gap[2]

        for resi in range(start_residue, end_residue + 1):
            resi_str = str(resi)
            resi_selection = f"protein2 and chain {chain} and resi {resi_str}"
            if cmd.count_atoms(resi_selection) > 0:
                new_residue_name = f"gap_residue_{resi_str}"
                cmd.create(new_residue_name, resi_selection)
                cmd.alter(new_residue_name, f"resi='{resi_str}'")
                cmd.alter(new_residue_name, f"chain='{chain}'")
                filled_residues.append(new_residue_name)

    # Save the modified structure
    cmd.create("filled_protein", "protein1 or " + " or ".join(filled_residues))
    cmd.save("filled_structure_raw.pdb", "filled_protein")

# Example usage
pdb_file1 = "experiment.pdb"  # Replace with your experimental PDB file
pdb_file2 = "alphafold_renumbered.pdb"  # Replace with your AlphaFold PDB file

gaps = find_gaps(pdb_file1)

if gaps:
    for gap in gaps:
        print(f"Gap detected in chain {gap[2]} between residues {gap[0]} and {gap[1]}")
    fill_gaps(gaps, pdb_file2)
else:
    print("No gaps detected")

# Cleanup
cmd.delete("all")

def reorder_pdb(input_pdb, output_pdb):
    with open(input_pdb, 'r') as file:
        lines = file.readlines()
    
    # Parse the ATOM lines to extract residues
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    residue_dict = {}
    
    for line in atom_lines:
        chain = line[21]
        resi = int(line[22:26].strip())
        if chain not in residue_dict:
            residue_dict[chain] = {}
        if resi not in residue_dict[chain]:
            residue_dict[chain][resi] = []
        residue_dict[chain][resi].append(line)
    
    # Sort the residues and write to the new file
    with open(output_pdb, 'w') as file:
        for chain in residue_dict:
            sorted_resis = sorted(residue_dict[chain].keys())
            for resi in sorted_resis:
                for line in residue_dict[chain][resi]:
                    # Clean column 72 by setting it to ' ' (empty)
                    line = line[:72] + ' ' + line[73:]
                    file.write(line)
        
        # Write the other lines (like TER, HETATM, etc.)
        other_lines = [line for line in lines if not line.startswith("ATOM")]
        for line in other_lines:
            file.write(line)

    # Remove extra TER lines at the end of the file
    with open(output_pdb, 'r') as file:
        lines = file.readlines()
    
    with open(output_pdb, 'w') as file:
        for line in lines:
            if line.startswith("TER") and lines.index(line) == len(lines) - 1:
                continue
            file.write(line)

# Reorder the residues in the PDB file
reorder_pdb("filled_structure_raw.pdb", "filled_structure.pdb")


