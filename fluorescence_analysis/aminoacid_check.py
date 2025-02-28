import os
import requests
import numpy as np
from Bio import PDB


def download_alphafold_pdb(uniprot_id, output_dir="pdb_files"):
    """Download the latest AlphaFold structure for a given UniProt ID."""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{uniprot_id}.pdb")

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"Failed to download PDB for {uniprot_id}")
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
                    if abs(omega) < 50:  # Cis (~0°)
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


def main(uniprot_ids):
    """Main function to analyze a list of UniProt IDs."""
    results = {}
    for uniprot_id in uniprot_ids:
        print(f"Processing {uniprot_id}...")
        pdb_file = download_alphafold_pdb(uniprot_id)
        if pdb_file:
            proline_data = analyze_prolines(pdb_file)
            buried_data = analyze_buried_residues(pdb_file)
            results[uniprot_id] = {"prolines": proline_data, "buried_residues": buried_data}

    print("\nSummary:")
    for uniprot_id, data in results.items():
        print(f"{uniprot_id}: {data['prolines']['cis']} cis, {data['prolines']['trans']} trans")
        print(
            f"    TRP - {data['buried_residues']['TRP']['buried']} buried, {data['buried_residues']['TRP']['exposed']} exposed")
        print(
            f"    TYR - {data['buried_residues']['TYR']['buried']} buried, {data['buried_residues']['TYR']['exposed']} exposed")


if __name__ == "__main__":
    #uniprot_list = ["E4THH1", "F8E4N3", "A0A2N2F632" ]  # Example UniProt IDs
    uniprot_list = ["A0A352WGM4", "A8SPL3" ,"A0A7X4IB97", "A0A540WI11", "A0A147GCS9", "A0A1G7WDE6", "A0A7D7NGB3", "A0A653LYW1", "A0A1V1V225"]  # Example UniProt IDs
    main(uniprot_list)
