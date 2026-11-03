#!/usr/bin/env python3
"""
calculate_secondary_structure_native_pdb.py

Oblicza ułamki struktur drugorzędowych (DSSP) dla dedykowanych struktur krystalicznych:
- CnTrmD dimer (PDB ID: 8BYH)
- CnTm1570 dimer (PDB ID: 8RI0)
- CnTrmD-Tm1570 fuzja dimer (PDB ID: 8B1N)
- FsTrmD-Tm1570 monomer (AlphaFold)
"""

import os
import urllib.request
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.DSSP import DSSP

PDB_TARGETS = {
    "8BYH": "8byh.pdb",
    "8RI0": "8ri0.pdb",
    "8B1N": "8b1n.pdb"
}


def download_pdbs():
    pdbl = PDBList()
    for code, fname in PDB_TARGETS.items():
        if not os.path.exists(fname):
            print(f"[+] Pobieranie PDB {code}...")
            pdbl.retrieve_pdb_file(code, pdir=".", file_format="pdb")
            downloaded = f"pdb{code.lower()}.ent"
            if os.path.exists(downloaded):
                os.rename(downloaded, fname)


def run_dssp_analysis(pdb_path, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    try:
        dssp = DSSP(model, pdb_path)
    except Exception as e:
        print(f"[-] Błąd DSSP dla {pdb_path}: {e}")
        return None

    counts = {"alpha": 0, "beta": 0, "coil": 0, "total": 0}

    for key in dssp.keys():
        c_id = key[0]
        if chain_id and c_id != chain_id:
            continue

        ss = dssp[key][2]
        counts["total"] += 1

        # Klasyfikacja DSSP:
        # H (alpha-helix), G (3_10-helix), I (pi-helix) -> Helisy
        # E (extended strand), B (beta-bridge)          -> Pasma beta
        # T (turn), S (bend), '-' (loop/coil)           -> Coil
        if ss in ["H", "G", "I"]:
            counts["alpha"] += 1
        elif ss in ["E", "B"]:
            counts["beta"] += 1
        else:
            counts["coil"] += 1

    tot = counts["total"]
    if tot == 0:
        return None

    return {
        "N_res": tot,
        "f_alpha": counts["alpha"] / tot,
        "f_beta": counts["beta"] / tot,
        "f_coil": counts["coil"] / tot,
        "n_alpha": counts["alpha"],
        "n_beta": counts["beta"]
    }


def main():
    download_pdbs()

    configs = [
        {"name": "CnTrmD (PDB: 8BYH, Chain A)", "file": "8byh.pdb", "chain": "A"},
        {"name": "CnTrmD (PDB: 8BYH, Dimer A+B)", "file": "8byh.pdb", "chain": None},
        {"name": "CnTm1570 (PDB: 8RI0, Chain A)", "file": "8ri0.pdb", "chain": "A"},
        {"name": "CnTm1570 (PDB: 8RI0, Dimer A+B)", "file": "8ri0.pdb", "chain": None},
        {"name": "CnTrmD-Tm1570 (PDB: 8B1N, Chain A)", "file": "8b1n.pdb", "chain": "A"},
        {"name": "CnTrmD-Tm1570 (PDB: 8B1N, Dimer A+B)", "file": "8b1n.pdb", "chain": None},
    ]

    # Dodanie pliku AlphaFold dla Fs (jeśli istnieje lokalnie)
    for af_candidate in ["AF-F8E4N3-F1-model_v6.pdb", "AF-F8E4N3-F1-model_v6.pdb", "Fs_AlphaFold.pdb"]:
        if os.path.exists(af_candidate):
            configs.append({"name": "FsTrmD-Tm1570 (AlphaFold Monomer)", "file": af_candidate, "chain": "A"})
            break

    print("\n" + "=" * 80)
    print(f"{'Konstrukt / Struktura':<35} | {'N_res':<6} | {'f_alpha':<8} | {'f_beta':<8} | {'f_coil':<8}")
    print("=" * 80)

    for c in configs:
        if not os.path.exists(c["file"]):
            continue
        res = run_dssp_analysis(c["file"], chain_id=c["chain"])
        if res:
            print(
                f"{c['name']:<35} | {res['N_res']:<6} | {res['f_alpha']:<8.3f} | {res['f_beta']:<8.3f} | {res['f_coil']:<8.3f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()