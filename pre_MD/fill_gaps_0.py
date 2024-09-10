from Bio.PDB import PDBParser, PDBIO, PDBExceptions, Superimposer
from Bio.PDB import Structure, Model, Chain, Residue
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys

def find_gaps(pdb_file):
    gaps = []

    # Initialize PDB parser
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("structure", pdb_file)
    except PDBExceptions.PDBConstructionException as e:
        print(f"Error parsing {pdb_file}: {e}")
        return []

    # Iterate over model, chain, and residue to find gaps
    for model in structure:
        for chain in model:
            prev_residue_id = None
            for residue in chain:
                residue_id = residue.get_id()[1]  # Get residue number
                if prev_residue_id is not None and residue_id != prev_residue_id + 1:
                    gaps.append((prev_residue_id + 1, residue_id - 1, chain.id))
                prev_residue_id = residue_id

    return gaps

def align_structures(structure1, structure2):
    ref_atoms = []
    alt_atoms = []

    for chain1 in structure1.get_chains():
        chain2 = structure2[0][chain1.id]
        for residue1, residue2 in zip(chain1, chain2):
            if residue1.id[0] == ' ' and residue2.id[0] == ' ':
                ref_atoms.append(residue1['CA'])
                alt_atoms.append(residue2['CA'])

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)
    super_imposer.apply(structure2.get_atoms())

    return structure2

def renumber_structure(structure, start_residue_id):
    for model in structure:
        for chain in model:
            new_id = start_residue_id
            for residue in chain:
                residue.id = (' ', new_id, ' ')
                new_id += 1
    return structure

def fill_gaps(gaps, structure2, structure1):
    filled_residues = {chain.id: [] for chain in structure1.get_chains()}

    for gap in gaps:
        start_residue = gap[0]
        end_residue = gap[1]
        chain_id = gap[2]

        chain = structure2[0][chain_id]  # Get chain from structure2

        for residue in chain:
            residue_id = residue.get_id()[1]
            if start_residue <= residue_id <= end_residue:
                filled_residues[chain_id].append(residue)

    return filled_residues

def save_filled_structure(filled_residues, original_structure, output_file):
    # Create a new structure object
    new_structure = Structure.Structure(original_structure.id)
    model = Model.Model(0)
    new_structure.add(model)

    # Add chains and residues to the new structure
    for chain in original_structure.get_chains():
        new_chain = Chain.Chain(chain.id)
        model.add(new_chain)

        residues_to_add = list(chain) + filled_residues.get(chain.id, [])
        residues_to_add.sort(key=lambda r: r.id[1])  # Sort residues by their ID

        for residue in residues_to_add:
            new_chain.add(residue)

    # Write filled structure to a new PDB file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fill_gaps.py <experiment_pdb_file> <alphafold_pdb_file> <output_pdb_file>")
        sys.exit(1)

    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Initialize PDB parser
    parser = PDBParser(QUIET=True)

    try:
        original_structure = parser.get_structure("original_structure", pdb_file1)
    except PDBExceptions.PDBConstructionException as e:
        print(f"Error parsing {pdb_file1}: {e}")
        sys.exit(1)

    try:
        alphafold_structure = parser.get_structure("alphafold_structure", pdb_file2)
    except PDBExceptions.PDBConstructionException as e:
        print(f"Error parsing {pdb_file2}: {e}")
        sys.exit(1)

    # Align AlphaFold structure to the original structure
    aligned_structure = align_structures(original_structure, alphafold_structure)

    # Renumber aligned AlphaFold structure to match the original structure
    start_residue_id = list(original_structure.get_residues())[0].get_id()[1]
    renumbered_structure = renumber_structure(aligned_structure, start_residue_id)

    # Find gaps in the experimental PDB file
    gaps = find_gaps(pdb_file1)

    if gaps:
        for gap in gaps:
            print(f"Gap detected in chain {gap[2]} between residues {gap[0]} and {gap[1]}")

        # Fill gaps using sequences from the aligned and renumbered AlphaFold PDB file
        filled_residues = fill_gaps(gaps, renumbered_structure, original_structure)

        if filled_residues:
            # Save the filled structure to a new PDB file
            save_filled_structure(filled_residues, original_structure, output_file)
            print(f"Filled structure saved to {output_file}")
        else:
            print("Failed to fill gaps. Check error messages above.")
    else:
        print("No gaps detected in the input PDB file.")

