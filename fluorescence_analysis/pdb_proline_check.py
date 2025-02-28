import os
import requests
import numpy as np
from Bio import PDB

def download_pdb_file(pdb_id, output_dir="pdb_files"):
    """Download a PDB file from the RCSB PDB database."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"Failed to download PDB for {pdb_id}")
        return None

def calculate_omega_angle(residue1, residue2):
    """Calculate the omega dihedral angle (C-N-Cα-C) between two residues."""
    try:
        atoms = [
            residue1["C"].get_vector(),
            residue2["N"].get_vector(),
            residue2["CA"].get_vector(),
            residue2["C"].get_vector()
        ]
        return PDB.calc_dihedral(*atoms) * 180.0 / np.pi  # Convert radians to degrees
    except KeyError:
        return None

def analyze_prolines(pdb_file):
    """Analyze cis vs. trans isomerism of prolines in a PDB structure."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # First model

    proline_states = {"cis": 0, "trans": 0}

    for chain in model:
        residues = list(chain.get_residues())
        for i in range(len(residues) - 1):
            res1, res2 = residues[i], residues[i + 1]
            if res2.get_resname() == "PRO":
                omega = calculate_omega_angle(res1, res2)
                if omega is not None:
                    if abs(omega) < 53:  # Cis (~0°)
                        proline_states["cis"] += 1
                    else:  # Trans (~180°)
                        proline_states["trans"] += 1

    return proline_states

def analyze_buried_residues(pdb_file):
    """Analyze the number of buried and exposed tryptophans (TRP) and tyrosines (TYR)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # First model

    buried_residues = {"TRP": {"buried": 0, "exposed": 0}, "TYR": {"buried": 0, "exposed": 0}}
    sasa_calculator = PDB.ShrakeRupley()
    sasa_calculator.compute(model, level="R")

    for chain in model:
        for residue in chain.get_residues():
            if residue.get_resname() in ["TRP", "TYR"]:
                sasa = residue.sasa
                if sasa < 10:  # Arbitrary cutoff for buried vs. exposed
                    buried_residues[residue.get_resname()]["buried"] += 1
                else:
                    buried_residues[residue.get_resname()]["exposed"] += 1

    return buried_residues

def main(pdb_ids):
    """Main function to analyze a list of PDB IDs."""
    results = {}
    for pdb_id in pdb_ids:
        print(f"Processing {pdb_id}...")
        pdb_file = download_pdb_file(pdb_id)
        if pdb_file:
            proline_data = analyze_prolines(pdb_file)
            buried_data = analyze_buried_residues(pdb_file)
            results[pdb_id] = {"prolines": proline_data, "buried_residues": buried_data}

    print("\nSummary:")
    for pdb_id, data in results.items():
        print(f"{pdb_id}: {data['prolines']['cis']} cis, {data['prolines']['trans']} trans")
        print(
            f"    TRP - {data['buried_residues']['TRP']['buried']} buried, {data['buried_residues']['TRP']['exposed']} exposed")
        print(
            f"    TYR - {data['buried_residues']['TYR']['buried']} buried, {data['buried_residues']['TYR']['exposed']} exposed")

if __name__ == "__main__":
    pdb_list = ["8B1N", "8RI0", "8BYH"]  # Example PDB IDs
    main(pdb_list)