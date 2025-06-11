import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from datetime import timedelta
from io import StringIO
from scipy.optimize import curve_fit

# === User Settings ===
input_csv = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/spectra_kinetics/60h/8_3_zeta_spectra_kin_60h00000.csv"
native_spectrum_path = None#"/home/matifortunka/Documents/JS/data_Cambridge/8_3/A/spectra_kinetics/8_3_A_5uM_nat00043_raw.txt"
dead_time = 120  # seconds
nm_per_sec = 0.1

path = "/".join(input_csv.split('/')[:-1])
output_plot = f"{path}/Combined_CD_HHMM.png"
hv_threshold = 1000
smoothing_window = 11
smoothing_polyorder = 3

# Plot 2
target_wavelength = 225.0
fit_model = "double"

# Plot 3
integration_range = (190, 250)
integration_sign = "negative"

# Baseline
baseline_correction = True
baseline_wavelength = 250.0

# Manual transpose
transpose_data = False

print("\n🔧 Parameters:")
print(f"  input_csv = {input_csv}")
print(f"  native_spectrum_path = {native_spectrum_path}")

# === Helpers ===
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

def is_float_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def print_fit_params(popt, pcov, param_names):
    print("Fitted parameters:")
    for i, name in enumerate(param_names):
        error = np.sqrt(np.diag(pcov))[i] if pcov is not None else float('nan')
        print(f"  {name} = {popt[i]:.8f} ± {error:.8f}")

# === Load CSV ===
with open(input_csv, 'r') as f:
    lines = f.readlines()

cd_lines = extract_section(lines, "CircularDichroism")
hv_lines = extract_section(lines, "HV")

cd_df = pd.read_csv(StringIO(''.join(cd_lines)), skipinitialspace=True)
hv_df = pd.read_csv(StringIO(''.join(hv_lines)), skipinitialspace=True)
# Rename first column to "Wavelength"
cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)

# Drop completely empty columns
cd_df.dropna(axis=1, how='all', inplace=True)

# Build col_map safely, skipping 'Wavelength'
cd_col_map = {
    float(col): col
    for col in cd_df.columns
    if col != "Wavelength" and not col.startswith("Unnamed")
}
hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)
hv_df.dropna(axis=1, how='all', inplace=True)
hv_col_map = {
    float(col): col
    for col in hv_df.columns
    if col != "Wavelength" and not col.startswith("Unnamed")
}

print("📋 CD usable columns:", list(cd_col_map.keys()))

if transpose_data:
    print("🔄 Transposing CD and HV dataframes (special reshaping)")

    # --- Transpose CD ---
    cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce')
    cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce')
    cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
    cd_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    cd_df = cd_df.dropna(subset=['Wavelength'])  # drop NaN wavelengths
    cd_df = cd_df.sort_values('Wavelength').reset_index(drop=True)

    wavelengths = cd_df['Wavelength'].values

    # --- Transpose HV ---
    hv_times = pd.to_numeric(hv_df.iloc[:, 0], errors='coerce')
    hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors='coerce')
    hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    hv_df = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times).reset_index()
    hv_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    hv_df = hv_df.dropna(subset=['Wavelength'])  # drop NaN wavelengths here too
    hv_df = hv_df.sort_values('Wavelength').reset_index(drop=True)

    # --- Align wavelength axes ---
    common_wavelengths = np.intersect1d(cd_df['Wavelength'].values, hv_df['Wavelength'].values)

    cd_df = cd_df[cd_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)
    hv_df = hv_df[hv_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)
    wavelengths = cd_df['Wavelength'].values  # update global wavelengths again

    # --- Time mappings ---
    cd_col_map = {float(col): col for col in cd_df.columns if col != 'Wavelength'}
    hv_col_map = {float(col): col for col in hv_df.columns if col != 'Wavelength'}

else:
    cd_col_map = {float(col): col for col in cd_df.columns[1:]}
    hv_col_map = {float(col): col for col in hv_df.columns[1:]}

cd_times = sorted(cd_col_map.keys())
hv_times = sorted(hv_col_map.keys())
wavelengths = cd_df.iloc[:, 0].values

# === Native spectrum ===
native_wl, native_cd = None, None
if native_spectrum_path:
    native_data = pd.read_csv(native_spectrum_path, sep='\t')
    native_wl = native_data.iloc[:, 0].values
    native_cd = native_data.iloc[:, 1].values

# === Time shift for plotting ===
time_shift = dead_time + (baseline_wavelength - target_wavelength) / nm_per_sec
shifted_cd_times = [t + time_shift for t in cd_times]
shifted_cd_times_hours = [t / 3600 for t in shifted_cd_times]
cd_times_hours = [t / 3600 for t in cd_times]

# === Plot 1: CD spectra ===
print("📈 Plot 1: CD spectra vs time")
plt.figure(figsize=(12, 8))
norm = Normalize(vmin=min(shifted_cd_times_hours), vmax=max(shifted_cd_times_hours))
cmap = cm.viridis
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for idx, (cd_time, shifted_cd_time_hr) in enumerate(zip(cd_times, shifted_cd_times_hours)):
    colname = cd_col_map.get(cd_time)
    if colname is None:
        continue

    cd = cd_df[colname].values
    hv_time = min(hv_times, key=lambda t: abs(t - cd_time))
    hv_col = hv_col_map.get(hv_time)
    hv = hv_df[hv_col].values

    if len(hv) != len(wavelengths):
        hv = hv[:len(wavelengths)]

    mask = hv <= hv_threshold
    x = wavelengths[mask]
    y = cd[mask]

    if len(y) >= smoothing_window:
        y = savgol_filter(y, window_length=smoothing_window, polyorder=smoothing_polyorder)

    if baseline_correction:
        baseline_idx = np.argmin(np.abs(x - baseline_wavelength))
        baseline_val = y[baseline_idx]
        y = y - baseline_val

    plt.plot(x, y, color=cmap(norm(shifted_cd_time_hr)))

# === Native spectrum smoothed and baseline corrected ===
if native_spectrum_path:
    native_y = native_cd.copy()
    if len(native_y) >= smoothing_window:
        native_y = savgol_filter(native_y, window_length=smoothing_window, polyorder=smoothing_polyorder)
    if baseline_correction:
        baseline_idx = np.argmin(np.abs(native_wl - baseline_wavelength))
        baseline_val = native_y[baseline_idx]
        native_y = native_y - baseline_val
    plt.plot(native_wl, native_y, color='black', linestyle='--', label='Native (0 s)')

plt.xlabel("Wavelength [nm]")
plt.ylabel("Ellipticity [mdeg]")
plt.title(f"CD Kinetics (HV ≤ {hv_threshold} V, Savitzky-Golay)")
plt.colorbar(sm, label="Time [h]", ax=plt.gca(), format="%.1f")
#plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot)
plt.show()
print(f"✅ Plot 1 saved: {output_plot}")

# === Plot 2: CD at specific wavelength ===
print("📉 Plot 2: CD at specific wavelength")
closest_idx = np.argmin(np.abs(wavelengths - target_wavelength))
if np.isnan(wavelengths[closest_idx]):
    print("❌ Error: Target wavelength not found (NaN in wavelengths)")
    actual_wavelength = target_wavelength
else:
    actual_wavelength = wavelengths[closest_idx]

from math import log
extra_dead_time = (baseline_wavelength - actual_wavelength) / nm_per_sec
effective_dead_time = dead_time + extra_dead_time

print(f"ℹ️ Effective dead time for {actual_wavelength} nm: {effective_dead_time:.1f} s")

cd_values_at_wl = []
valid_times_s = []
valid_times_hr = []

if native_spectrum_path and native_wl is not None:
    native_y = native_cd.copy()
    if len(native_y) >= smoothing_window:
        native_y = savgol_filter(native_y, window_length=smoothing_window, polyorder=smoothing_polyorder)
    if baseline_correction:
        baseline_idx = np.argmin(np.abs(native_wl - baseline_wavelength))
        baseline_val = native_y[baseline_idx]
        native_y = native_y - baseline_val
    native_idx = np.argmin(np.abs(native_wl - actual_wavelength))
    native_val = native_y[native_idx]
    cd_values_at_wl.append(native_val)
    valid_times_s.append(effective_dead_time)
    valid_times_hr.append(effective_dead_time / 3600)

for cd_time in cd_times:
    cd_col = cd_col_map.get(cd_time)
    hv_time = min(hv_col_map.keys(), key=lambda t: abs(t - cd_time))
    hv_col = hv_col_map.get(hv_time)

    cd_curve = cd_df[cd_col].values
    hv_val = hv_df[hv_col].values[closest_idx]

    if hv_val <= hv_threshold:
        if len(cd_curve) >= smoothing_window:
            cd_curve = savgol_filter(cd_curve, window_length=smoothing_window, polyorder=smoothing_polyorder)
        if baseline_correction:
            baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
            baseline_val = cd_curve[baseline_idx]
            cd_val = cd_curve[closest_idx] - baseline_val
        else:
            cd_val = cd_curve[closest_idx]
        cd_values_at_wl.append(cd_val)
        adjusted_time = cd_time + effective_dead_time
        valid_times_s.append(adjusted_time)
        valid_times_hr.append(adjusted_time / 3600)

plt.figure(figsize=(6, 5))
plt.plot(valid_times_hr, cd_values_at_wl, marker='o', color='darkred', label="Data")
plt.xlabel("Time [h]", fontsize=16)
plt.ylabel(f"Ellipticity at {target_wavelength} [mdeg]", fontsize=16)
#plt.title(f"CD at {actual_wavelength:.1f} nm vs Time")
#plt.grid(True)

# === Fit in seconds ===
def linear(t, k, b): return k * t + b
def single_exp(t, a, k, c): return a * np.exp(-k * t) + c
def double_exp(t, a, k1, c, k2, e): return a * np.exp(-k1 * t) + c * np.exp(-k2 * t) + e

t_fit = np.array(valid_times_s)
y_fit = np.array(cd_values_at_wl)

if len(valid_times_s) < 5:
    print(f"⚠️  Not enough points for fitting: {len(valid_times_s)} point(s)")
    fit_model = None

try:
    if fit_model == "linear":
        popt, pcov = curve_fit(linear, t_fit, y_fit)
        print_fit_params(popt, pcov, ['k', 'b'])
        plt.plot(valid_times_hr, linear(t_fit, *popt), linestyle='--', label='Linear Fit')
    elif fit_model == "single":
        popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[-1]), maxfev=5000)
        print_fit_params(popt, pcov, ['a', 'k', 'c'])
        t_half = log(2) / popt[1]
        print(f"🧮 Half-life (t₁/₂) = {t_half:.2f} s = {t_half / 3600:.2f} h")
        plt.plot(valid_times_hr, single_exp(t_fit, *popt), linestyle='--', label='Single Exp Fit')
    elif fit_model == "double":
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[0]/2, 0.0001, y_fit[-1]), maxfev=5000)
        print_fit_params(popt, pcov, ['a', 'k1', 'c', 'k2', 'e'])
        t_half1 = log(2) / popt[1]
        t_half2 = log(2) / popt[3]
        print(f"🧮 Half-lives (t₁/₂):")
        print(f"   Fast phase (k1): {t_half1:.2f} s = {t_half1 / 3600:.2f} h")
        print(f"   Slow phase (k2): {t_half2:.2f} s = {t_half2 / 3600:.2f} h")
        plt.plot(valid_times_hr, double_exp(t_fit, *popt), linestyle='--', label='Double Exp Fit')
except Exception as e:
    print(f"❌ Fit failed: {repr(e)}")

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
try:
    plt.savefig(f"{path}/CD_at_{int(actual_wavelength)}_nm_vs_time.png")
except Exception as e:
    print(f"❌ Failed to save Plot 2: {e}")
plt.show()


# === Plot 3: Integrated CD ===
print("📊 Plot 3: Integrated CD")
lambda_min, lambda_max = integration_range
sign = integration_sign.lower()
in_range_mask = (wavelengths >= lambda_min) & (wavelengths <= lambda_max)

integrated_cd_values = []
integration_times_s = []
integration_times_hr = []

mid_wavelength = 0.5 * (lambda_min + lambda_max)
extra_dead_time = (baseline_wavelength - mid_wavelength) / nm_per_sec
effective_dead_time = dead_time + extra_dead_time

print(f"ℹ️ Effective dead time for integration center ({mid_wavelength:.1f} nm): {effective_dead_time:.1f} s")

# === Native spectrum integration ===
if native_spectrum_path and native_wl is not None:
    x_nat = native_wl[(native_wl >= lambda_min) & (native_wl <= lambda_max)]
    y_nat = native_cd[(native_wl >= lambda_min) & (native_wl <= lambda_max)]

    if len(y_nat) >= smoothing_window:
        y_nat = savgol_filter(y_nat, window_length=smoothing_window, polyorder=smoothing_polyorder)

    if baseline_correction:
        baseline_idx = np.argmin(np.abs(x_nat - baseline_wavelength))
        baseline_val = y_nat[baseline_idx]
        y_nat = y_nat - baseline_val

    y_nat = np.where((y_nat > 0) if sign == "positive" else (y_nat < 0), y_nat, 0)

    if len(x_nat) > 1:
        area = trapezoid(y_nat, x_nat)
        integrated_cd_values.append(area)
        integration_times_s.append(effective_dead_time)
        integration_times_hr.append(effective_dead_time / 3600)

# === Time series integration ===
for cd_time in cd_times:
    cd_col = cd_col_map.get(cd_time)
    hv_time = min(hv_col_map.keys(), key=lambda t: abs(t - cd_time))
    hv_col = hv_col_map.get(hv_time)

    cd = cd_df[cd_col].values
    hv = hv_df[hv_col].values
    mask = hv <= hv_threshold
    if len(wavelengths) != len(hv):
        raise ValueError(f"Mismatch: wavelengths={len(wavelengths)} vs hv={len(hv)}")
    x = wavelengths[mask]
    in_range_mask_local = (x >= lambda_min) & (x <= lambda_max)
    x = x[in_range_mask_local]
    y = cd[mask][in_range_mask_local]

    if len(y) >= smoothing_window:
        y = savgol_filter(y, window_length=smoothing_window, polyorder=smoothing_polyorder)

    if baseline_correction and len(x) > 0:
        baseline_idx = np.argmin(np.abs(x - baseline_wavelength))
        baseline_val = y[baseline_idx]
        y = y - baseline_val

    y = np.where((y > 0) if sign == "positive" else (y < 0), y, 0)

    if len(x) > 1:
        area = trapezoid(y, x)
        time_s = cd_time + effective_dead_time
        integrated_cd_values.append(area)
        integration_times_s.append(time_s)
        integration_times_hr.append(time_s / 3600)

# === Plot + Fit ===
plt.figure(figsize=(10, 5))
plt.plot(integration_times_hr, integrated_cd_values, marker='o', color='blue', label="Data")
plt.xlabel("Time [h]", fontsize=16)
plt.ylabel("Integrated CD [mdeg·nm]", fontsize=16)
plt.title(f"Integrated {sign} CD ({lambda_min}-{lambda_max} nm) vs Time")
#plt.grid(True)

t_fit = np.array(integration_times_s)  # seconds
y_fit = np.array(integrated_cd_values)

try:
    if fit_model == "linear":
        popt, pcov = curve_fit(linear, t_fit, y_fit)
        print_fit_params(popt, pcov, ['k', 'b'])
        plt.plot(integration_times_hr, linear(t_fit, *popt), linestyle='--', label='Linear Fit')
    elif fit_model == "single":
        popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[-1]), maxfev=5000)
        print_fit_params(popt, pcov, ['a', 'k', 'c'])
        t_half = log(2) / popt[1]
        print(f"🧮 Half-life (t₁/₂) = {t_half:.2f} s = {t_half / 3600:.2f} h")
        plt.plot(integration_times_hr, single_exp(t_fit, *popt), linestyle='--', label='Single Exp Fit')
    elif fit_model == "double":
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[0]/2, 0.0001, y_fit[-1]), maxfev=5000)
        print_fit_params(popt, pcov, ['a', 'k1', 'c', 'k2', 'e'])
        t_half1 = log(2) / popt[1]
        t_half2 = log(2) / popt[3]
        print(f"🧮 Half-lives (t₁/₂):")
        print(f"   Fast phase (k1): {t_half1:.2f} s = {t_half1 / 3600:.2f} h")
        print(f"   Slow phase (k2): {t_half2:.2f} s = {t_half2 / 3600:.2f} h")
        plt.plot(integration_times_hr, double_exp(t_fit, *popt), linestyle='--', label='Double Exp Fit')
except Exception as e:
    print(f"❌ Fit failed: {repr(e)}")

plt.legend()
plt.tight_layout()
plt.savefig(f"{path}/CD_integrated_{sign}_{lambda_min}_{lambda_max}_vs_time.png")
plt.show()

