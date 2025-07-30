# === Final CD Analysis with SESCA Deconvolution and Optional Transpose Support ===

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import subprocess
from tempfile import NamedTemporaryFile
from io import StringIO
import json

# === Settings ===
sesca_script = "/home/matifortunka/Programs/SESCA/scripts/SESCA_deconv.py"
basis_set_code = "/home/matifortunka/Programs/SESCA/libs/Map_DS-dT.dat"
input_csv = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/A/spectra_kinetics/36h/8_3 alfa urea_unf_5 uM_time00003.csv"
output_path = os.path.dirname(input_csv)
output_plot = os.path.join(output_path, "Combined_CD_HHMM.png")
hv_threshold = 1000
smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = 250.0
dead_time = 420
nm_per_sec = 0.1
fit_model = None
transpose_data = False  # Set to True if data is transposed
verbose = False

# === CD unit conversion parameters ===
protein_conc = 5      # µmol/L (µM)
path_length_mm = 1          # mm
residue_count = 479         # number of residues per molecule -1

MRW = 113.4811273
conc = 0.8153619

# Convert from mdeg to mean residue ellipticity (deg·cm²·dmol⁻¹)
conversion_factor = 1000000 / (protein_conc * path_length_mm * residue_count)
# conversion_factor = 0.1 * MRW / (path_length_cm * conc)

# === Fit functions ===
def linear(t, k, b): return k * t + b
def single_exp(t, a, k, c): return a * np.exp(-k * t) + c
def double_exp(t, a, k1, c, k2, e): return a * np.exp(-k1 * t) + c * np.exp(-k2 * t) + e

def print_fit_params(popt, pcov, param_names):
    print("Fitted parameters:")
    for i, name in enumerate(param_names):
        error = np.sqrt(np.diag(pcov))[i] if pcov is not None else float('nan')
        print(f"  {name} = {popt[i]:.8f} ± {error:.8f}")

# === Load CSV and Extract Sections ===
def extract_section(lines, section_name):
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    else:
        raise ValueError(f"Section {section_name} not found")
    section_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        section_lines.append(line)
    return section_lines

with open(input_csv, 'r') as f:
    lines = f.readlines()

cd_df = pd.read_csv(StringIO(''.join(extract_section(lines, "CircularDichroism"))), skipinitialspace=True)
hv_df = pd.read_csv(StringIO(''.join(extract_section(lines, "HV"))), skipinitialspace=True)

cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
cd_df.dropna(axis=1, how='all', inplace=True)
hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)
hv_df.dropna(axis=1, how='all', inplace=True)

if transpose_data:
    print("🔄 Transposing CD and HV dataframes (special reshaping)")

    # --- Transpose CD ---
    cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce')
    cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce')
    cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
    cd_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    cd_df = cd_df.dropna(subset=['Wavelength'])
    cd_df = cd_df.sort_values('Wavelength').reset_index(drop=True)

    # --- Transpose HV ---
    hv_times = pd.to_numeric(hv_df.iloc[:, 0], errors='coerce')
    hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors='coerce')
    hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    hv_df = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times).reset_index()
    hv_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    hv_df = hv_df.dropna(subset=['Wavelength'])
    hv_df = hv_df.sort_values('Wavelength').reset_index(drop=True)

    # --- Align wavelength axes ---
    common_wavelengths = np.intersect1d(cd_df['Wavelength'].values, hv_df['Wavelength'].values)
    cd_df = cd_df[cd_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)
    hv_df = hv_df[hv_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)

    wavelengths = cd_df['Wavelength'].values

    cd_col_map = {float(col): col for col in cd_df.columns if col != 'Wavelength'}
    hv_col_map = {float(col): col for col in hv_df.columns if col != 'Wavelength'}

else:
    cd_col_map = {float(col): col for col in cd_df.columns if col != "Wavelength" and not col.startswith("Unnamed")}
    hv_col_map = {float(col): col for col in hv_df.columns if col != "Wavelength" and not col.startswith("Unnamed")}
    wavelengths = cd_df["Wavelength"].values

cd_times = sorted(cd_col_map.keys())
hv_times = sorted(hv_col_map.keys())

# === CD Plot ===
time_shift = dead_time
shifted_cd_times = [t + time_shift for t in cd_times]
shifted_cd_times_hours = [t / 3600 for t in shifted_cd_times]

plt.figure(figsize=(12, 8))
norm = Normalize(vmin=min(shifted_cd_times_hours), vmax=max(shifted_cd_times_hours))
cmap = cm.viridis
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for cd_time, shifted_time_hr in zip(cd_times, shifted_cd_times_hours):
    colname = cd_col_map[cd_time]
    y = cd_df[colname].values
    y = savgol_filter(y, smoothing_window, smoothing_polyorder)
    baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
    y -= y[baseline_idx]
    plt.plot(wavelengths, y, color=cmap(norm(shifted_time_hr)))

plt.xlabel("Wavelength [nm]")
plt.ylabel("Ellipticity [mdeg]")
plt.title("CD Spectra vs Time")
plt.colorbar(sm, label="Time [h]", ax=plt.gca(), format="%.1f")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot)
plt.show()

# === SESCA Deconvolution ===
checkpoint_file = os.path.join(output_path, "SESCA_structure_content.json")

if os.path.exists(checkpoint_file):
    print("✅ Loading SESCA checkpoint...")
    with open(checkpoint_file, 'r') as f:
        structure_content = json.load(f)
else:
    print("🔄 Running SESCA deconvolution...")
    structure_content = {"Time_hr": [], "alpha": [], "beta": [], "coil": []}

    for cd_time in cd_times:
        colname = cd_col_map[cd_time]
        y = cd_df[colname].values
        y = savgol_filter(y, smoothing_window, smoothing_polyorder)
        y -= y[np.argmin(np.abs(wavelengths - baseline_wavelength))]

        with NamedTemporaryFile(delete=False, suffix=".txt", mode="w", dir=output_path) as f:
            spectrum_path = f.name
            for wl, val in zip(wavelengths, y):
                val = val * conversion_factor
                f.write(f"{wl:.1f}\t{val:.6f}\n")

        try:
            result = subprocess.run(
                ["python3", sesca_script, spectrum_path, basis_set_code],
                capture_output=True,
                text=True,
                cwd=output_path
            )

            output_file = os.path.join(output_path, "BS_deconv.out")
            alpha = beta = coil = None
            try:
                with open(output_file, "r") as f_out:
                    for line in f_out:
                        if "Alpha" in line:
                            alpha = float(line.split(":")[-1].strip())
                        elif "Beta" in line:
                            beta = float(line.split(":")[-1].strip())
                        elif "Coil" in line:
                            coil = float(line.split(":")[-1].strip())
            except FileNotFoundError:
                print(f"❌ Output file not found: {output_file}")
            else:
                if None not in (alpha, beta, coil):
                    structure_content["Time_hr"].append(cd_time / 3600)
                    structure_content["alpha"].append(alpha)
                    structure_content["beta"].append(beta)
                    structure_content["coil"].append(coil)
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)

        except Exception as e:
            print(f"❌ SESCA execution error for t={cd_time}s: {e}")
        finally:
            os.remove(spectrum_path)

    with open(checkpoint_file, 'w') as f:
        json.dump(structure_content, f, indent=2)
    print("✅ SESCA results saved to checkpoint.")

# === Plot and Fit Structure Content ===
df_struct = pd.DataFrame(structure_content)
if df_struct.empty:
    print("❌ No structure data collected from SESCA.")
    exit(1)

plt.figure(figsize=(6, 5))

for label in ["alpha", "beta", "coil"]:
    t_fit = np.array(df_struct["Time_hr"]) * 3600
    y_fit = np.array(df_struct[label])
    try:
        if fit_model == "double":
            popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[0]/2, 0.0001, y_fit[-1]), maxfev=5000)
            print(f"\nFit for {label} (double exp):")
            print_fit_params(popt, pcov, ['a', 'k1', 'c', 'k2', 'e'])
            t_half_1 = np.log(2) / popt[1]
            t_half_2 = np.log(2) / popt[3]
            print(f"  t_half_1 = {t_half_1:.2f} s ({t_half_1/3600:.2f} h)")
            print(f"  t_half_2 = {t_half_2:.2f} s ({t_half_2/3600:.2f} h)")
            y_model = double_exp(t_fit, *popt)
            plt.plot(df_struct["Time_hr"], y_model, linestyle='--', label=f"{label} Fit")
        elif fit_model == "single":
            popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[-1]), maxfev=5000)
            print(f"\nFit for {label} (single exp):")
            print_fit_params(popt, pcov, ['a', 'k', 'c'])
            t_half = np.log(2) / popt[1]
            print(f"  t_half = {t_half:.2f} s ({t_half/3600:.2f} h)")
            y_model = single_exp(t_fit, *popt)
            plt.plot(df_struct["Time_hr"], y_model, linestyle='--', label=f"{label} Fit")
        elif fit_model is None:
            pass
        else:
            raise ValueError(f"Unsupported fit_model: {fit_model}")

    except Exception as e:
        print(f"⚠️ Fit failed for {label}: {e}")
    plt.scatter(df_struct["Time_hr"], df_struct[label], label=label)

plt.xlabel("Time [h]", fontsize=16)
plt.ylabel("Secondary structure content", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig(f"{output_path}/SESCA_structure_content_vs_time.png", dpi=600)
plt.savefig(f"{output_path}/SESCA_structure_content_vs_time.svg", dpi=600)
plt.show()
