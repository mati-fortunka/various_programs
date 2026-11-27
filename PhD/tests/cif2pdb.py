from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO


def convert_cif_to_pdb(input_dir: str = ".", output_dir: str = "pdb_output") -> None:
  in_path = Path(input_dir)
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  parser = MMCIFParser(QUIET=True)
  io = PDBIO()

  # Adjust the filename pattern if your files are named differently,
  # e.g., f"fold_2af94bf3b5666c37_model_{i}.cif"
  for i in range(5):
    cif_file = in_path / f"/home/matifortunka/Downloads/PhD/rozdział_IV/fold_f8e4n3_fstrmd_tm1570_dimer_uniprot_seq/fold_f8e4n3_fstrmd_tm1570_dimer_uniprot_seq_model_{i}.cif"
    if not cif_file.exists():
      print(f"Skipping {cif_file} (not found)")
      continue

    structure = parser.get_structure(f"model_{i}", str(cif_file))
    pdb_file = out_path / f"model_{i}.pdb"

    io.set_structure(structure)
    io.save(str(pdb_file))
    print(f"Converted: {cif_file} -> {pdb_file}")


if __name__ == "__main__":
  convert_cif_to_pdb(input_dir=".", output_dir="pdb_output")