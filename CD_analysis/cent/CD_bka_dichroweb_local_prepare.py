import os
import re
import numpy as np
from scipy.signal import savgol_filter

# === Konfiguracja ===
input_folder = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/urea/tm_unfold_SAV"
output_folder = os.path.join(input_folder, "dichrowebgit_input")
os.makedirs(output_folder, exist_ok=True)

# Parametry białka Tm1570
protein_conc_uM = 20.0  # µmol/L
path_length_mm = 0.1  # mm (0.01 cm)
residue_count = 224

# Wybór jednostki docelowej:
# Jeśli A.txt w SP175 ma wartości rzędu -5 do +5 -> wybierz "delta_epsilon"
# Jeśli A.txt w SP175 ma wartości rzędu -10000 do -30000 -> wybierz "mre"
TARGET_UNIT = "delta_epsilon"

# Przeliczniki:
# MRE [deg*cm^2/dmol] = (mdeg * 100) / (conc_uM * path_mm * residues)
# Delta_epsilon [M^-1 * cm^-1] = theta_mdeg / (32980 * conc_M * path_cm) = (theta_mdeg * 1000) / (3298 * conc_uM * path_mm * residues)
if TARGET_UNIT == "delta_epsilon":
  unit_factor = 1000.0 / (3298.0 * protein_conc_uM * path_length_mm)
else:
  unit_factor = 1000.0 / (protein_conc_uM * path_length_mm * residue_count)

min_wl = 190.0  # pobieramy pełen zakres, w GUI ustawisz odcięcie (np. 200 nm)
max_wl = 245.0


def extract_conc(filename):
  m = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?:\.[^.]+)?$", filename)
  if m:
    return float(m.group(1))
  name = filename.lower()
  if "native" in name or "0m" in name or "buffer" in name:
    return 0.0
  return None


files_created = 0

for fname in sorted(os.listdir(input_folder)):
  if not fname.endswith(".bka"):
    continue

  conc = extract_conc(fname)
  if conc is None:
    continue

  filepath = os.path.join(input_folder, fname)
  with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

  try:
    start_idx = (
        next(
            i
            for i, line in enumerate(lines)
            if line.strip().strip('"') == "_DATA"
        )
        + 1
    )
  except StopIteration:
    continue

  wls, vals = [], []
  for line in lines[start_idx:]:
    l = line.strip()
    if not l or not any(c.isdigit() for c in l):
      break
    parts = l.split()
    if len(parts) >= 2:
      try:
        wls.append(float(parts[0]))
        vals.append(float(parts[1]))
      except ValueError:
        continue

  wls = np.array(wls)
  vals = np.array(vals)

  # Wygładzenie Savitzky-Golay
  vals_smooth = savgol_filter(vals, 11, 3)

  # Przeliczenie jednostek
  vals_converted = vals_smooth * unit_factor

  # Dopasowanie zakresu fal
  mask = (wls >= min_wl) & (wls <= max_wl)
  wls_sel = wls[mask]
  vals_sel = vals_converted[mask]

  # Zapewnienie stałego kroku 1.0 nm i kolejności rosnącej/malejącej
  sort_idx = np.argsort(wls_sel)[::-1]
  wls_sorted = wls_sel[sort_idx]
  vals_sorted = vals_sel[sort_idx]

  # Zapis tab-delimited (wavelength \t value)
  out_name = f"Tm1570_{conc:.1f}M.txt"
  out_path = os.path.join(output_folder, out_name)

  with open(out_path, "w", encoding="utf-8") as f_out:
    for w, v in zip(wls_sorted, vals_sorted):
      f_out.write(f"{w:.1f}\t{v:.5f}\n")

  files_created += 1

print(
    f"✅ Przygotowano {files_created} plików widm w formacie dwukolumnowym (tab-delimited)"
)
print(f"📁 Zapisano w: {output_folder}")