from Bio import PDB
import sys

def renumber_residues(pdb_file, output_file, offset, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    old_id = residue.id
                    new_id = (old_id[0], old_id[1] + offset, old_id[2])
                    residue.id = new_id
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python renumber_chain.py <input_pdb_file> <output_pdb_file> <offset> <chain_id>")
        sys.exit(1)

    input_pdb_file = sys.argv[1]
    output_pdb_file = sys.argv[2]
    offset = int(sys.argv[3])
    chain_id = sys.argv[4]

    renumber_residues(input_pdb_file, output_pdb_file, offset, chain_id)
    print(f"Residue numbering in chain {chain_id} adjusted by {offset} and saved to {output_pdb_file}")

