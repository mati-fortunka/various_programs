from Bio.PDB import PDBParser, PDBIO, PDBExceptions, Superimposer
from Bio.PDB import Structure, Model, Chain, Residue
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
import sys

def extract_sequence(chain):
    seq = []
    for residue in chain:
        if residue.id[0] == ' ':
            seq.append(residue.resname)
    return ''.join(seq)

def align_chains(chain1, chain2):
    seq1 = extract_sequence(chain1)
    seq2 = extract_sequence(chain2)
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    return alignments[0]

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
    chain_mappings = {}

    for chain1 in structure1.get_chains():
        best_chain = None
        best_alignment_score = -1

        for chain2 in structure2.get_chains():
            alignment = align_chains(chain1, chain2)
            score = alignment.score / max(len(chain1), len(chain2))
            if score > best_alignment_score:
                best_alignment_score = score
                best_chain = chain2

        if best_chain:
            chain_mappings[chain1.id] = best_chain.id
            ref_atoms = []
            alt_atoms = []

            for residue1, residue2 in zip(chain1, best_chain):
                if residue1.id[0] == ' ' and residue2.id[0] == ' ':
                    ref_atoms.append(residue1['CA'])
                    alt_atoms.append(residue2['CA'])

            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_atoms, alt_atoms)
            super_imposer.apply(structure2.get_atoms())

    return structure2, chain_mappings

def find_next_unique_residue_id(existing_ids, target_id):
    new_id = target_id
    increment = 1
    while new_id in existing_ids:
        new_id = (target_id[0], target_id[1] + increment, target_id[2])
        increment += 1
    return new_id

def renumber_structure(structure, reference_structure, chain_mappings):
    ref_residues = {chain.id: list(chain) for chain in reference_structure.get_chains()}
    existing_ids = set()
    for chain in reference_structure.get_chains():
        for residue in chain:
            existing_ids.add(residue.id)

    for model in structure:
        for chain in model:
            if chain.id in chain_mappings:
                ref_chain_id = chain_mappings[chain.id]
                ref_chain_residues = ref_residues[ref_chain_id]
                for i, residue in enumerate(chain):
                    if i < len(ref_chain_residues):
                        try:
                            residue.id = ref_chain_residues[i].id
                        except ValueError:
                            # Generate a unique ID for the residue
                            new_id = find_next_unique_residue_id(existing_ids, ref_chain_residues[i].id)
                            residue.id = new_id
                            existing_ids.add(new_id)
    return structure

def fill_gaps(gaps, structure2, structure1, chain_mappings):
    filled_residues = {chain.id: [] for chain in structure1.get_chains()}

    for gap in gaps:
        start_residue = gap[0]
        end_residue = gap[1]
        chain_id = gap[2]

        if chain_id in chain_mappings:
            aligned_chain_id = chain_mappings[chain_id]
            chain = structure2[0][aligned_chain_id]

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
    aligned_structure, chain_mappings = align_structures(original_structure, alphafold_structure)

    # Renumber aligned AlphaFold structure to match the original structure
    renumbered_structure = renumber_structure(aligned_structure, original_structure, chain_mappings)

    # Find gaps in the experimental PDB file
    gaps = find_gaps(pdb_file1)

    if gaps:
        for gap in gaps:
            print(f"Gap detected in chain {gap[2]} between residues {gap[0]} and {gap[1]}")

        # Fill gaps using sequences from the aligned and renumbered AlphaFold PDB file
        filled_residues = fill_gaps(gaps, renumbered_structure, original_structure, chain_mappings)

        if filled_residues:
            # Save the filled structure to a new PDB file
            save_filled_structure(filled_residues, original_structure, output_file)
            print(f"Filled structure saved to {output_file}")
        else:
            print("Failed to fill gaps. Check error messages above.")
    else:
        print("No gaps detected in the input PDB file.")

