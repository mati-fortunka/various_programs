# === Final CD Analysis with SVD Deconvolution and Optional Transpose Support ===

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from io import StringIO

# === Settings ===
input_csv = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/spectra_kinetics/60h/8_3_zeta_spectra_kin_60h00000.csv"
output_path = os.path.dirname(input_csv)
output_plot = os.path.join(output_path, "Combined_CD_HHMM.png")
hv_threshold = 1000
smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = 250.0
dead_time = 120
nm_per_sec = 0.1
fit_model = None
transpose_data = False  # Set to True if data is transposed

# === Fit functions ===
def double_exp(t, a, k1, c, k2, e): return a * np.exp(-k1 * t) + c * np.exp(-k2 * t) + e
def single_exp(t, a, k, c): return a * np.exp(-k * t) + c

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
    cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce')
    cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce')
    cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
    cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
    cd_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    cd_df = cd_df.dropna(subset=['Wavelength'])
    cd_df = cd_df.sort_values('Wavelength').reset_index(drop=True)

    hv_times = pd.to_numeric(hv_df.iloc[:, 0], errors='coerce')
    hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors='coerce')
    hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
    hv_df = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times).reset_index()
    hv_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    hv_df = hv_df.dropna(subset=['Wavelength'])
    hv_df = hv_df.sort_values('Wavelength').reset_index(drop=True)

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

# === SVD-Based Structure Estimation ===
print("🔍 Performing SVD on CD spectra for structure estimation")

cd_matrix = np.column_stack([
    savgol_filter(cd_df[col].values - cd_df[col].values[np.argmin(np.abs(wavelengths - baseline_wavelength))],
                  smoothing_window, smoothing_polyorder)
    for _, col in sorted(cd_col_map.items())
])

U, S, VT = np.linalg.svd(cd_matrix, full_matrices=False)
components = U.T @ cd_matrix
components = components[:3, :]  # top 3 components
fractions = np.maximum(components, 0)
fractions = fractions / np.sum(fractions, axis=0, keepdims=True)

structure_content = {
    "Time_hr": [t / 3600 for t in cd_times],
    "alpha": fractions[1],
    "beta": fractions[2],
    "coil": fractions[0]
}

# === Plot and Fit Structure Content ===
df_struct = pd.DataFrame(structure_content)

plt.figure(figsize=(6, 5))
for label in ["alpha", "beta", "coil"]:
    t_fit = np.array(df_struct["Time_hr"]) * 3600
    y_fit = np.array(df_struct[label])
    try:
        if fit_model == "double":
            popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[0] / 2, 0.0001, y_fit[-1]), maxfev=5000)
            print(f"\nDouble Exponential Fit for {label}:")
            print_fit_params(popt, pcov, ['a', 'k1', 'c', 'k2', 'e'])
            plt.plot(df_struct["Time_hr"], double_exp(t_fit, *popt), linestyle='--', label=f"{label} Double Fit")
        elif fit_model == "single":
            popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[-1]), maxfev=5000)
            print(f"\nSingle Exponential Fit for {label}:")
            print_fit_params(popt, pcov, ['a', 'k', 'c'])
            plt.plot(df_struct["Time_hr"], single_exp(t_fit, *popt), linestyle='--', label=f"{label} Single Fit")

    except Exception as e:
        print(f"⚠️ Fit failed for {label}: {e}")
    plt.plot(df_struct["Time_hr"], df_struct[label], marker='o', label=label.capitalize())

plt.xlabel("Time [h]", fontsize=16)
plt.ylabel("Secondary structure content", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=14, frameon=False)
#plt.title("Secondary Structure Content Over Time (SVD Deconv)")
#plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_path}/SVD_structure_content_vs_time.png", dpi = 600)
plt.show()
