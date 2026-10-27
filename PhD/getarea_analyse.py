#!/usr/bin/env python3
"""parse_getarea_clean.py

Oczyszczony parser plikow GETAREA:
- eksportuje wylacznie fluorofory aromatyczne (TRP i TYR),
- bezpiecznie sumuje SASA globalne (z obsluga NaN w Fs).
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV/"
GETAREA_FILES = [
    {
        "filename": "Cnfusion_getarea.txt",
        "construct": "CnTrmD-Tm1570",
        "state": "Homodimer (Chain A/B)",
        "source": "PDB 8B1N",
    },
    {
        "filename": "Cnfusion_monomer_getarea.txt",
        "construct": "CnTrmD-Tm1570",
        "state": "Fusion Monomer",
        "source": "PDB 8B1N",
    },
    {
        "filename": "TrmD_monomer_getarea.txt",
        "construct": "CnTrmD",
        "state": "Isolated Monomer",
        "source": "PDB 8B1N (1-240)",
    },
    {
        "filename": "Tm1570_monomer_getarea.txt",
        "construct": "CnTm1570",
        "state": "Isolated Monomer",
        "source": "PDB 8B1N (241-430)",
    },
    {
        "filename": "FSfusion_getarea.txt",
        "construct": "FsTrmD-Tm1570",
        "state": "Fusion Monomer",
        "source": "AlphaFold",
    },
]

OUTPUT_DIR = "tables"
CSV_FLUOROPHORES = os.path.join(OUTPUT_DIR, "getarea_aromatic_fluorophores.csv")
CSV_GLOBAL_SASA = os.path.join(OUTPUT_DIR, "getarea_global_sasa_summary.csv")


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
      if len(parts) >= 6:
        resn = parts[0]
        resi = int(parts[1]) if parts[1].lstrip("-").isdigit() else parts[1]

        def to_float(val):
          try:
            return float(val)
          except ValueError:
            return np.nan

        tot = to_float(parts[2])
        apol = to_float(parts[3])
        bb = to_float(parts[4])
        sc = to_float(parts[5])
        ratio = to_float(parts[6])
        in_out = parts[7] if len(parts) >= 8 else "undecided"

        if in_out == "undecided":
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

    if "POLAR  area/energy" in line:
      val = line.split("=")[-1].strip()
      global_stats["polar"] = float(val) if val != "NaN" else np.nan
    elif "APOLAR area/energy" in line:
      val = line.split("=")[-1].strip()
      global_stats["apolar"] = float(val) if val != "NaN" else np.nan
    elif "Total  area/energy" in line:
      val = line.split("=")[-1].strip()
      global_stats["total"] = float(val) if val != "NaN" else np.nan
    elif "Number of surface atoms" in line:
      val = line.split("=")[-1].strip()
      global_stats["surface_atoms"] = int(val) if val.isdigit() else np.nan
    elif "Number of buried atoms" in line:
      val = line.split("=")[-1].strip()
      global_stats["buried_atoms"] = int(val) if val.isdigit() else np.nan

  df_res = pd.DataFrame(residues)

  # Jesli Total z GETAREA to NaN (jak w Fs), wylicz sume bezposrednio z reszt
  if np.isnan(global_stats["total"]) and not df_res.empty:
    global_stats["total"] = np.round(np.nansum(df_res["total_sasa"]), 2)
    global_stats["apolar"] = np.round(np.nansum(df_res["apolar_sasa"]), 2)

  return df_res, global_stats


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  all_fluorophores = []
  global_summary = []

  for item in GETAREA_FILES:
    fname = INPUT_DIR+item["filename"]
    if not os.path.exists(fname):
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

    # SCIŚLE: Tylko reszty aromatyczne (TRP i TYR)
    arom = df_res[df_res["resname"].isin(["TRP", "TYR"])].copy()
    arom["Construct"] = item["construct"]
    arom["State"] = item["state"]
    arom["Source"] = item["source"]

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

    arom["Module"] = arom.apply(assign_module, axis=1)
    all_fluorophores.append(arom)

  # Zapis podsumowania globalnego
  pd.DataFrame(global_summary).to_csv(CSV_GLOBAL_SASA, index=False)
  print(f"[OK] Zapisano: {CSV_GLOBAL_SASA}")

  # Zapis tabeli fluoroforow
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