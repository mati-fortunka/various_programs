#!/usr/bin/env python3
"""
calculate_secondary_structure_native_pdb.py

Oblicza ułamki struktur drugorzędowych (DSSP) dla dedykowanych struktur:
- CnTrmD dimer (PDB ID: 8BYH)
- CnTm1570 dimer (PDB ID: 8RI0)
- CnTrmD-Tm1570 fuzja dimer (PDB ID: 8B1N)
- FsTrmD-Tm1570 monomer i dimer (AlphaFold)

oraz eksportuje podsumowanie do pliku CSV.
"""

import os
import tempfile
import warnings
import pandas as pd
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.DSSP import DSSP

PDB_TARGETS = {
    "8BYH": "8byh.pdb",
    "8RI0": "8ri0.pdb",
    "8B1N": "8b1n.pdb",
}

FS_DIMER_PDB = "/home/matifortunka/PycharmProjects/various_programs/PhD/pdb_output/model_0.pdb"
OUTPUT_DIR = "tables"
CSV_DSSP_SUMMARY = os.path.join(OUTPUT_DIR, "dssp_secondary_structure_summary.csv")


def download_pdbs():
    pdbl = PDBList()
    for code, fname in PDB_TARGETS.items():
        if not os.path.exists(fname):
            print(f"[+] Pobieranie PDB {code}...")
            pdbl.retrieve_pdb_file(code, pdir=".", file_format="pdb")
            downloaded = f"pdb{code.lower()}.ent"
            if os.path.exists(downloaded):
                os.rename(downloaded, fname)


def _prepare_dssp_input(pdb_path):
    """
    Zapewnia, że pliki PDB bez nagłówka HEADER (np. po konwersji przez PDBIO)
    są poprawnie rozpoznawane przez mkdssp jako format PDB.
    Zwraca (ścieżka_do_pliku, czy_tymczasowy).
    """
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    # Pliki mmCIF zaczynają się zwykle od 'data_', a standardowe PDB od 'HEADER'
    if first_line.startswith("data_") or first_line.startswith("HEADER"):
        return pdb_path, False

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8")
    try:
        tmp.write("HEADER    GENERATED FOR DSSP                                                \n")
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as src:
            tmp.write(src.read())
        tmp.close()
        return tmp.name, True
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


def run_dssp_analysis(pdb_path, chains=None):
    """
    chains:
      - None: wszystkie łańcuchy w modelu
      - str: pojedynczy łańcuch, np. "A"
      - set/list/tuple: zbiór łańcuchów, np. {"A", "B"} dla dimeru
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    if isinstance(chains, str):
        allowed_chains = {chains}
    elif chains is not None:
        allowed_chains = set(chains)
    else:
        allowed_chains = None

    dssp_input_path, is_temp = _prepare_dssp_input(pdb_path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
            dssp = DSSP(model, dssp_input_path)
    except Exception as e:
        print(f"[-] Błąd DSSP dla {pdb_path}: {e}")
        return None
    finally:
        if is_temp and os.path.exists(dssp_input_path):
            os.remove(dssp_input_path)

    counts = {"alpha": 0, "beta": 0, "coil": 0, "total": 0}

    for key in dssp.keys():
        c_id = key[0]
        if allowed_chains is not None and c_id not in allowed_chains:
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
        "f_alpha": round(counts["alpha"] / tot, 4),
        "f_beta": round(counts["beta"] / tot, 4),
        "f_coil": round(counts["coil"] / tot, 4),
        "n_alpha": counts["alpha"],
        "n_beta": counts["beta"],
        "n_coil": counts["coil"],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_pdbs()

    configs = [
        {"name": "CnTrmD (PDB: 8BYH, Chain A)", "file": "8byh.pdb", "chains": "A"},
        {"name": "CnTrmD (PDB: 8BYH, Dimer A+B)", "file": "8byh.pdb", "chains": {"A", "B"}},
        {"name": "CnTm1570 (PDB: 8RI0, Chain A)", "file": "8ri0.pdb", "chains": "A"},
        {"name": "CnTm1570 (PDB: 8RI0, Dimer A+B)", "file": "8ri0.pdb", "chains": {"A", "B"}},
        {"name": "CnTrmD-Tm1570 (PDB: 8B1N, Chain A)", "file": "8b1n.pdb", "chains": "A"},
        {"name": "CnTrmD-Tm1570 (PDB: 8B1N, Dimer A+B)", "file": "8b1n.pdb", "chains": {"A", "B"}},
    ]

    # Dodanie pliku monomeru AlphaFold dla Fs (jeśli istnieje lokalnie)
    for af_candidate in ["AF-F8E4N3-F1-model_v6.pdb", "Fs_AlphaFold.pdb"]:
        if os.path.exists(af_candidate):
            configs.append({
                "name": "FsTrmD-Tm1570 (AF Monomer, Chain A)",
                "file": af_candidate,
                "chains": "A",
            })
            break

    # Dodanie dimeru Fs z model_0.pdb
    if os.path.exists(FS_DIMER_PDB):
        configs.append({
            "name": "FsTrmD-Tm1570 (AF3 m0, Chain A)",
            "file": FS_DIMER_PDB,
            "chains": "A",
        })
        configs.append({
            "name": "FsTrmD-Tm1570 (AF3 m0, Dimer A+B)",
            "file": FS_DIMER_PDB,
            "chains": {"A", "B"},
        })
    else:
        print(f"[WARN] Nie znaleziono pliku dimeru Fs: {FS_DIMER_PDB}")

    rows = []

    print("\n" + "=" * 86)
    print(f"{'Konstrukt / Struktura':<44} | {'N_res':<6} | {'f_alpha':<8} | {'f_beta':<8} | {'f_coil':<8}")
    print("=" * 86)

    for c in configs:
        if not os.path.exists(c["file"]):
            continue
        res = run_dssp_analysis(c["file"], chains=c["chains"])
        if res:
            print(
                f"{c['name']:<44} | {res['N_res']:<6} | {res['f_alpha']:<8.3f} | {res['f_beta']:<8.3f} | {res['f_coil']:<8.3f}"
            )
            chain_label = (
                "ALL"
                if c["chains"] is None
                else c["chains"]
                if isinstance(c["chains"], str)
                else "+".join(sorted(c["chains"]))
            )
            rows.append({
                "Structure": c["name"],
                "Chains": chain_label,
                "N_res": res["N_res"],
                "n_alpha": res["n_alpha"],
                "n_beta": res["n_beta"],
                "n_coil": res["n_coil"],
                "f_alpha": res["f_alpha"],
                "f_beta": res["f_beta"],
                "f_coil": res["f_coil"],
                "File": os.path.abspath(c["file"]),
            })

    print("=" * 86 + "\n")

    if rows:
        pd.DataFrame(rows).to_csv(CSV_DSSP_SUMMARY, index=False)
        print(f"[OK] Zapisano: {CSV_DSSP_SUMMARY}")


if __name__ == "__main__":
    main()