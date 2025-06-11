import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import subprocess
from tempfile import NamedTemporaryFile
import shutil

# === Configuration ===
sesca_script = "/home/matifortunka/Programs/SESCA/scripts/SESCA_deconv.py"
basis_set_code = "/home/matifortunka/Programs/SESCA/libs/Map_DS-dT.dat"
input_csv = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/sav-golay3m.bka"  # CHANGE THIS
smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = 250.0
verbose = False

# === Load spectrum ===
df = pd.read_csv(input_csv, sep="\t")  # assume tab-separated
df.columns = ["Wavelength", "Ellipticity"]
#df = df.sort_values(by="Wavelength")
#df.reset_index(drop=True, inplace=True)
wavelengths = df["Wavelength"].values
ellipticity = df["Ellipticity"].values

# === Smooth and baseline correct ===
ellipticity_smooth = savgol_filter(ellipticity, smoothing_window, smoothing_polyorder)
baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
ellipticity_corrected = ellipticity_smooth - ellipticity_smooth[baseline_idx]

# === Plot corrected spectrum ===
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, ellipticity_corrected, label="Corrected Ellipticity")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Ellipticity [mdeg]")
plt.title("Corrected CD Spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()

base, _ = os.path.splitext(input_csv)
plot_path = base + "_corrected.png"
plt.savefig(plot_path)
plt.close()

# === Write spectrum to temporary file for SESCA ===
output_dir = os.path.dirname(input_csv)
spectrum_txt = base + "_temp_sesca.txt"
final_output = base + "_deconv.out"

with open(spectrum_txt, "w") as f:
    for wl, val in zip(wavelengths, ellipticity_corrected):
        f.write(f"{wl:.1f}\t{val:.6f}\n")

# === Run SESCA ===
try:
    result = subprocess.run(
        ["python3", sesca_script, spectrum_txt, basis_set_code],
        capture_output=True,
        text=True,
        cwd=output_dir
    )

    if verbose:
        print("=== SESCA STDOUT ===\n", result.stdout)
        print("=== SESCA STDERR ===\n", result.stderr)

    sesca_out = os.path.join(output_dir, "BS_deconv.out")
    final_output = input_csv.replace(".txt", "_deconv.out")
    final_output = input_csv.replace(".bka", "_deconv.out")

    alpha = beta = coil = None
    if os.path.exists(sesca_out):
        shutil.copy(sesca_out, final_output)
        with open(sesca_out, "r") as f:
            for line in f:
                if "Alpha" in line:
                    alpha = float(line.split(":")[-1].strip())
                elif "Beta" in line:
                    beta = float(line.split(":")[-1].strip())
                elif "Coil" in line:
                    coil = float(line.split(":")[-1].strip())

        os.remove(sesca_out)

        print("\n✅ SESCA Deconvolution Results:")
        print(f"  Alpha content: {alpha:.4f}")
        print(f"  Beta content:  {beta:.4f}")
        print(f"  Coil/Other:    {coil:.4f}")
        print(f"\n📝 Saved output to: {final_output}")
    else:
        print("❌ SESCA output file not found!")

except Exception as e:
    print(f"❌ SESCA execution error: {e}")

finally:
    if os.path.exists(spectrum_txt):
        os.remove(spectrum_txt)
