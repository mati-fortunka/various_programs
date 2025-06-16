import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import subprocess
import shutil

# === Configuration ===
sesca_script = "/home/matifortunka/Programs/SESCA/scripts/SESCA_deconv.py"
basis_set_code = "/home/matifortunka/Programs/SESCA/libs/Map_DS-dT.dat"
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/unfolding_SAV"  # Folder containing .bka files
smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = 250.0
verbose = False

# === Data storage ===
concentrations = []
alpha_contents = []
beta_contents = []
coil_contents = []

# === Helper to extract concentration from filename ===
def extract_concentration(filename):
    match = re.search(r"([\d.]+)m", filename)
    return float(match.group(1)) if match else None

# === Process all .bka files ===
for file in os.listdir(input_folder):
    if not file.endswith(".bka"):
        continue

    input_path = os.path.join(input_folder, file)
    concentration = extract_concentration(file)
    if concentration is None:
        continue

    # Load and process spectrum, skipping headers
    with open(input_path, "r") as f:
        lines = f.readlines()
    try:
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
    except StopIteration:
        print(f"⚠️ No _DATA section in {file}")
        continue

    # Parse numeric data manually (robust to formatting issues)
    data = []
    for line in lines[data_start_idx:]:
        if line.strip() == "" or not any(c.isdigit() for c in line):
            break
        try:
            wavelength, value = map(float, line.strip().split())
            data.append((wavelength, value))
        except ValueError:
            continue
    if not data:
        print(f"⚠️ No valid data in {file}")
        continue

    df = pd.DataFrame(data, columns=["Wavelength", "Ellipticity"])
    wavelengths = df["Wavelength"].values
    ellipticity = df["Ellipticity"].values

    # Smooth and baseline correct
    ellipticity_smooth = savgol_filter(ellipticity, smoothing_window, smoothing_polyorder)
    baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
    ellipticity_corrected = ellipticity_smooth - ellipticity_smooth[baseline_idx]

    # Write to temporary file
    base = os.path.splitext(input_path)[0]
    spectrum_txt = base + "_temp_sesca.txt"
    final_output = base + "_deconv.out"

    with open(spectrum_txt, "w") as f:
        for wl, val in zip(wavelengths, ellipticity_corrected):
            f.write(f"{wl:.1f}\t{val:.6f}\n")

    # Run SESCA
    try:
        result = subprocess.run([
            "python3", sesca_script, spectrum_txt, basis_set_code
        ], capture_output=True, text=True, cwd=input_folder)

        sesca_out = os.path.join(input_folder, "BS_deconv.out")
        if os.path.exists(sesca_out):
            shutil.copy(sesca_out, final_output)
            with open(sesca_out, "r") as f:
                alpha = beta = coil = None
                for line in f:
                    if "Alpha" in line:
                        alpha = float(line.split(":")[-1].strip())
                    elif "Beta" in line:
                        beta = float(line.split(":")[-1].strip())
                    elif "Coil" in line:
                        coil = float(line.split(":")[-1].strip())
            os.remove(sesca_out)

            if None not in (alpha, beta, coil):
                concentrations.append(concentration)
                alpha_contents.append(alpha)
                beta_contents.append(beta)
                coil_contents.append(coil)

                print(f"Processed {file}: Alpha={alpha:.4f}, Beta={beta:.4f}, Coil={coil:.4f}")
            else:
                print(f"⚠️ Missing structure data in: {file}")

        else:
            print(f"❌ SESCA output missing for: {file}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

    finally:
        if os.path.exists(spectrum_txt):
            os.remove(spectrum_txt)

# === Plotting ===
if concentrations:
    conc = np.array(concentrations)
    sort_idx = np.argsort(conc)

    plt.figure(figsize=(10, 6))
    plt.plot(conc[sort_idx], np.array(alpha_contents)[sort_idx], label="Alpha", marker="o")
    plt.plot(conc[sort_idx], np.array(beta_contents)[sort_idx], label="Beta", marker="s")
    plt.plot(conc[sort_idx], np.array(coil_contents)[sort_idx], label="Coil", marker="^")
    plt.xlabel("Denaturant Concentration [M]")
    plt.ylabel("Structure Content")
    plt.title("Secondary Structure vs. Denaturant Concentration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(input_folder, "SESCA_structure_vs_concentration.png"))
    plt.show()
else:
    print("No valid results to plot.")
