#!/usr/bin/env python3
"""parse_getarea_clean.py

Oczyszczony parser plikow GETAREA:
- eksportuje wylacznie fluorofory aromatyczne (TRP i TYR),
- bezpiecznie sumuje SASA globalne (z obsluga NaN w Fs),
- uwzglednia dimer Fs (model_0.pdb) oraz monomer Fs z tego samego modelu (model_0_chA.pdb).
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV/"
FS_DIMER_PDB = "/home/matifortunka/PycharmProjects/various_programs/PhD/pdb_output/model_0.pdb"
FS_MONOMER_CHA_PDB = "/home/matifortunka/PycharmProjects/various_programs/PhD/pdb_output/model_0_chA.pdb"

FS_DIMER_GETAREA = os.path.join(INPUT_DIR, "Fsfusion_dimer_AF3_m0_getarea.txt")
# Upewnij się, że nazwa odpowiada wynikowi GETAREA dla model_0_chA.pdb:
FS_MONOMER_CHA_GETAREA = os.path.join(INPUT_DIR, "Fsfusion_dimer_AF3_m0_chA_getarea.txt")

GETAREA_FILES = [
    {
        "filepath": os.path.join(INPUT_DIR, "Cnfusion_getarea.txt"),
        "construct": "CnTrmD-Tm1570",
        "state": "Homodimer (Chain A/B)",
        "source": "PDB 8B1N",
    },
    {
        "filepath": os.path.join(INPUT_DIR, "Cnfusion_monomer_getarea.txt"),
        "construct": "CnTrmD-Tm1570",
        "state": "Fusion Monomer",
        "source": "PDB 8B1N",
    },
    {
        "filepath": os.path.join(INPUT_DIR, "TrmD_monomer_getarea.txt"),
        "construct": "CnTrmD",
        "state": "Isolated Monomer",
        "source": "PDB 8B1N (1-240)",
    },
    {
        "filepath": os.path.join(INPUT_DIR, "Tm1570_monomer_getarea.txt"),
        "construct": "CnTm1570",
        "state": "Isolated Monomer",
        "source": "PDB 8B1N (241-430)",
    },
    {
        "filepath": os.path.join(INPUT_DIR, "FSfusion_getarea.txt"),
        "construct": "FsTrmD-Tm1570",
        "state": "Fusion Monomer",
        "source": "AlphaFold DB",
    },
    {
        "filepath": FS_MONOMER_CHA_GETAREA,
        "construct": "FsTrmD-Tm1570",
        "state": "Fusion Monomer (Chain A)",
        "source": f"AlphaFold3 ({os.path.basename(FS_MONOMER_CHA_PDB)})",
    },
    {
        "filepath": FS_DIMER_GETAREA,
        "construct": "FsTrmD-Tm1570",
        "state": "Homodimer (Chain A/B)",
        "source": f"AlphaFold3 ({os.path.basename(FS_DIMER_PDB)})",
    },
]

OUTPUT_DIR = "tables"
CSV_FLUOROPHORES = os.path.join(OUTPUT_DIR, "getarea_aromatic_fluorophores.csv")
CSV_GLOBAL_SASA = os.path.join(OUTPUT_DIR, "getarea_global_sasa_summary.csv")


def ensure_chain_a_pdb(dimer_pdb=FS_DIMER_PDB, chain_a_pdb=FS_MONOMER_CHA_PDB, chain_id="A"):
    """Opcjonalnie wyodrębnia łańcuch A z pliku dimeru model_0.pdb do model_0_chA.pdb."""
    if os.path.exists(chain_a_pdb) or not os.path.exists(dimer_pdb):
        return

    os.makedirs(os.path.dirname(chain_a_pdb), exist_ok=True)
    with open(dimer_pdb, "r", encoding="utf-8", errors="ignore") as src, open(
        chain_a_pdb, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            rec = line[:6].strip()
            if rec in {"ATOM", "ANISOU", "TER"} and len(line) > 21 and line[21] == chain_id:
                dst.write(line)
            elif rec in {"HEADER", "TITLE", "CRYST1", "END"}:
                dst.write(line)


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def parse_getarea(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    residues = []
    global_stats = {
        "polar": np.nan,
        "apolar": np.nan,
        "total": np.nan,
        "surface_atoms": np.nan,
        "buried_atoms": np.nan,
    }

    in_table = False
    for line in lines:
        line_str = line.strip()
        norm_line = " ".join(line_str.split())

        if (
            "Residue" in line_str
            and "Total" in line_str
            and "Ratio(%)" in line_str
        ):
            in_table = True
            continue
        if line_str.startswith("---------------------------------------------"):
            in_table = False
            continue

        if in_table:
            parts = line_str.split()
            if len(parts) >= 7:
                resn = parts[0]
                resi = int(parts[1]) if parts[1].lstrip("-").isdigit() else parts[1]

                tot = to_float(parts[2])
                apol = to_float(parts[3])
                bb = to_float(parts[4])
                sc = to_float(parts[5])
                ratio = to_float(parts[6])
                in_out = parts[7] if len(parts) >= 8 else "undecided"

                if in_out == "undecided" and not np.isnan(ratio):
                    if ratio < 20.0:
                        in_out = "i"
                    elif ratio > 50.0:
                        in_out = "o"

                residues.append({
                    "resname": resn,
                    "resnum": resi,
                    "total_sasa": tot,
                    "apolar_sasa": apol,
                    "backbone_sasa": bb,
                    "sidechain_sasa": sc,
                    "ratio_percent": ratio,
                    "in_out": in_out,
                })

        if norm_line.startswith("POLAR area/energy"):
            val = norm_line.split("=")[-1].strip()
            global_stats["polar"] = np.nan if val.upper() == "NAN" else to_float(val)
        elif norm_line.startswith("APOLAR area/energy"):
            val = norm_line.split("=")[-1].strip()
            global_stats["apolar"] = np.nan if val.upper() == "NAN" else to_float(val)
        elif norm_line.startswith("Total area/energy"):
            val = norm_line.split("=")[-1].strip()
            global_stats["total"] = np.nan if val.upper() == "NAN" else to_float(val)
        elif "Number of surface atoms" in norm_line:
            val = norm_line.split("=")[-1].strip()
            global_stats["surface_atoms"] = int(val) if val.isdigit() else np.nan
        elif "Number of buried atoms" in norm_line:
            val = norm_line.split("=")[-1].strip()
            global_stats["buried_atoms"] = int(val) if val.isdigit() else np.nan

    df_res = pd.DataFrame(residues)

    # Jesli Total z GETAREA to NaN (jak w Fs), wylicz sume bezposrednio z reszt
    if np.isnan(global_stats["total"]) and not df_res.empty:
        global_stats["total"] = round(float(np.nansum(df_res["total_sasa"])), 2)
    if np.isnan(global_stats["apolar"]) and not df_res.empty:
        global_stats["apolar"] = round(float(np.nansum(df_res["apolar_sasa"])), 2)

    return df_res, global_stats


def assign_module(row):
    num = row["resnum"]
    if isinstance(num, int):
        if num <= 235:
            return "TrmD"
        elif 236 <= num <= 245:
            return "Linker"
        else:
            return "Tm1570"
    return "Unknown"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_chain_a_pdb()

    all_fluorophores = []
    global_summary = []

    for item in GETAREA_FILES:
        fname = item["filepath"]
        if not os.path.exists(fname):
            print(f"[WARN] Pominięto brakujący plik: {fname}")
            continue

        df_res, glob = parse_getarea(fname)

        global_summary.append({
            "Construct": item["construct"],
            "State": item["state"],
            "Source": item["source"],
            "Total_SASA_A2": glob["total"],
            "Polar_SASA_A2": glob["polar"],
            "Apolar_SASA_A2": glob["apolar"],
            "Surface_Atoms": glob["surface_atoms"],
            "Buried_Atoms": glob["buried_atoms"],
            "File": fname,
        })

        if not df_res.empty:
            # SCIŚLE: Tylko reszty aromatyczne (TRP i TYR)
            arom = df_res[df_res["resname"].isin(["TRP", "TYR"])].copy()
            arom["Construct"] = item["construct"]
            arom["State"] = item["state"]
            arom["Source"] = item["source"]
            arom["Module"] = arom.apply(assign_module, axis=1)
            all_fluorophores.append(arom)

    pd.DataFrame(global_summary).to_csv(CSV_GLOBAL_SASA, index=False)
    print(f"[OK] Zapisano: {CSV_GLOBAL_SASA}")

    if all_fluorophores:
        df_arom = pd.concat(all_fluorophores, ignore_index=True)
        cols = [
            "Construct",
            "State",
            "Module",
            "resname",
            "resnum",
            "ratio_percent",
            "in_out",
            "total_sasa",
            "sidechain_sasa",
            "Source",
        ]
        df_arom = df_arom[cols]
        df_arom.to_csv(CSV_FLUOROPHORES, index=False)
        print(f"[OK] Zapisano: {CSV_FLUOROPHORES}")
        print(f"Liczba wyekstrahowanych fluoroforow: {len(df_arom)}")


if __name__ == "__main__":
    main()