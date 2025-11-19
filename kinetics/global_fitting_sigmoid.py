import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from lmfit import Minimizer, Parameters, report_fit


# ==========================================
# 1. LOADERS
# ==========================================

def extract_section(lines, section_name):
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    else:
        return None
    section_lines = []
    for line in lines[start:]:
        if not line.strip(): break
        section_lines.append(line)
    return section_lines


def load_complex_cd_spectra(filename, target_wl=222, dead_time=400):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None, None

    print(f"Processing CD Spectra: {os.path.basename(filename)}")
    with open(filename, 'r') as f:
        lines = f.readlines()

    cd_lines = extract_section(lines, "CircularDichroism")
    if not cd_lines: return None, None

    cd_df = pd.read_csv(StringIO(''.join(cd_lines)), skipinitialspace=True)
    cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
    cd_df.dropna(axis=1, how='all', inplace=True)

    try:
        cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce')
        cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
        cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce')

        cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
        cd_df.rename(columns={'index': 'Wavelength'}, inplace=True)
        cd_df = cd_df.dropna(subset=['Wavelength']).sort_values('Wavelength').reset_index(drop=True)

        wavelengths = cd_df['Wavelength'].values
        closest_idx = np.argmin(np.abs(wavelengths - target_wl))

        times = np.array(
            [float(c) for c in cd_df.columns if isinstance(c, (int, float)) or str(c).replace('.', '', 1).isdigit()])
        times = np.sort(times)
        intensities = np.array([cd_df[t].values[closest_idx] for t in times])

        return times + dead_time, intensities
    except Exception as e:
        print(f"Error parsing CD: {e}")
        return None, None


def read_sf_data(filename):
    if not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            skiprows = 1 if not re.match(r'^[\d\.\-]', f.readline().strip().split(',')[0]) else 0

        df = pd.read_csv(filename, skiprows=skiprows, header=None, usecols=[0, 1]).dropna()
        df.columns = ['time', 'val']
        t = df['time'].values
        wrap_idx = np.where(np.diff(t) < 0)[0]
        if len(wrap_idx) > 0: df = df.iloc[:wrap_idx[0] + 1]
        return df['time'].values, df['val'].values
    except Exception as e:
        print(f"Error SF: {e}")
        return None, None


def get_all_files(folder, protein):
    search = 'dzeta' if protein == 'zeta' else protein
    files = []
    for f in os.listdir(folder):
        if f.endswith(".csv") and search in f:
            files.append(os.path.join(folder, f))
    return files


def cut_data(t, y, t_min=None, t_max=None):
    if t is None: return None, None
    mask = np.ones(len(t), dtype=bool)
    if t_min is not None: mask &= (t >= t_min)
    if t_max is not None: mask &= (t <= t_max)
    return t[mask], y[mask]


def decimate(t, y, max_pts=300):
    if t is None or len(t) <= max_pts: return t, y
    step = int(np.ceil(len(t) / max_pts))
    return t[::step], y[::step]


# ==========================================
# 2. HYBRID MODEL (Exp + Sigmoid)
# ==========================================

def hybrid_model(t, amps, rates, t_half_D, offset):
    """
    A, B, C, E, F = Exponential Decay
    D = Sigmoid (Logistic Function)
    """
    model = np.full_like(t, offset, dtype=float)

    # Exponentials (Indices 0,1,2 for A,B,C and 4,5 for E,F)
    exp_indices = [0, 1, 2, 4, 5]
    for i in exp_indices:
        exponent = np.clip(-rates[i] * t, -700, 700)
        model += amps[i] * np.exp(exponent)

    # Sigmoid for Phase D (Index 3)
    # rates[3] here acts as the STEEPNESS (k)
    if abs(amps[3]) > 1e-12:
        sigmoid_exponent = np.clip(-rates[3] * (t - t_half_D), -700, 700)
        model += amps[3] / (1 + np.exp(sigmoid_exponent))

    return model


def objective(params, data_blocks):
    residuals = []
    rates = [params[f'k_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    t_half_D = params['t_half_D']

    for block in data_blocks:
        t, y = block['t'], block['y']

        if block['type'] == 'SF':
            amps = [params[f'amp_sf_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
            offset = params[block['offset_key']]
        else:  # CD
            amps = [params[f'amp_cd_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
            offset = params['off_cd']

        model = hybrid_model(t, amps, rates, t_half_D, offset)

        # Weight CD higher
        weight = 5.0 if block['type'] == 'CD' else 1.0
        residuals.append((y - model) * weight)

    return np.concatenate(residuals)


# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":

    PROTEIN = "gamma"
    TARGET_WL = 222

    # --- PATHS ---
    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"
    paths = {
        'SF_Fast': os.path.join(base_sf, "phase_A"),
        'SF_Slow': os.path.join(base_sf, "double_exp_B-C")
    }
    cd_spectra_file = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"

    # --- CUT CONFIG ---
    CUT_CONFIG = {
        'SF_Fast': {'min': 0.002, 'max': None},
        'SF_Slow': {'min': 1.0, 'max': None},
        'CD': {'min': None, 'max': None}
    }

    data_blocks = []
    params = Parameters()

    # --- 1. Load SF ---
    for category, folder in paths.items():
        files = get_all_files(folder, PROTEIN)
        for i, fpath in enumerate(files):
            t_raw, y_raw = read_sf_data(fpath)
            if t_raw is not None:
                limits = CUT_CONFIG.get(category, {})
                t_cut, y_cut = cut_data(t_raw, y_raw, limits.get('min'), limits.get('max'))
                if t_cut is not None and len(t_cut) > 10:
                    t_dec, y_dec = decimate(t_cut, y_cut, 300)
                    offset_key = f"off_sf_{category}_{i}"
                    params.add(offset_key, value=np.mean(y_dec))
                    data_blocks.append({
                        't': t_dec, 'y': y_dec, 't_raw': t_cut, 'y_raw': y_cut,
                        'type': 'SF', 'name': f"{category}-{i}", 'offset_key': offset_key
                    })

    # --- 2. Load CD ---
    t_raw, y_raw = load_complex_cd_spectra(cd_spectra_file, target_wl=TARGET_WL, dead_time=30)
    if t_raw is not None:
        t_dec, y_dec = decimate(t_raw, y_raw, 300)
        data_blocks.append({'t': t_dec, 'y': y_dec, 't_raw': t_raw, 'y_raw': y_raw, 'type': 'CD', 'name': 'CD'})
        params.add('off_cd', value=y_dec[-1])

    if not data_blocks: print("No data loaded."); exit()

    # --- 3. Global Parameters ---
    # Exponentials
    params.add('k_A', value=np.log(2) / 0.38, min=0)
    params.add('k_B', value=np.log(2) / 12.0, min=0)
    params.add('k_C', value=np.log(2) / 334.0, min=0)

    # Phase D (Sigmoid)
    params.add('k_D', value=0.01, min=0)  # Steepness
    params.add('t_half_D', value=1500.0, min=1000, max=5000)  # Midpoint

    # Slow Exponentials
    params.add('k_E', value=np.log(2) / (2.4 * 3600), min=0)
    params.add('k_F', value=np.log(2) / (33 * 3600), min=0)

    # SF Amplitudes
    params.add('amp_sf_A', value=2.0)
    params.add('amp_sf_B', value=1.0)
    params.add('amp_sf_C', value=0.5)
    params.add('amp_sf_D', value=0.2)
    params.add('amp_sf_E', value=0, vary=False)
    params.add('amp_sf_F', value=0, vary=False)

    # CD Amplitudes
    params.add('amp_cd_A', value=0, vary=False)
    params.add('amp_cd_B', value=100)
    params.add('amp_cd_C', value=500)
    params.add('amp_cd_D', value=1000)
    params.add('amp_cd_E', value=2000)
    params.add('amp_cd_F', value=5000)

    # --- 4. Fit ---
    print(f"\nRunning Hybrid Global Fit (Exp + Sigmoid D)...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))
    result = minner.minimize(method='leastsq')

    # --- 5. Report (Modified) ---
    print("\n" + "=" * 40)
    print(f" RESULTS: {PROTEIN} (Hybrid Model)")
    print("=" * 40)

    four_hours = 4 * 3600  # 14400 seconds

    # Report Exponentials
    for p in ['A', 'B', 'C', 'E', 'F']:
        k = result.params[f'k_{p}'].value
        if k > 1e-10:
            th = np.log(2) / k

            # --- CUSTOM DISPLAY LOGIC ---
            if th <= four_hours:
                print(f"Phase {p} (Exp): {th:.2f} s  ({th / 3600:.3f} h)")
            else:
                print(f"Phase {p} (Exp): {th / 3600:.2f} h")

    # Report Sigmoid
    th_D = result.params['t_half_D'].value
    slope_D = result.params['k_D'].value

    # --- CUSTOM DISPLAY LOGIC FOR SIGMOID ---
    if th_D <= four_hours:
        print(f"Phase D (Sigmoid): {th_D:.2f} s  ({th_D / 3600:.3f} h) [Slope k={slope_D:.4f}]")
    else:
        print(f"Phase D (Sigmoid): {th_D / 3600:.2f} h [Slope k={slope_D:.4f}]")

    # --- 6. Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    rates = [result.params[f'k_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    t_half_D = result.params['t_half_D'].value
    sf_amps = [result.params[f'amp_sf_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]

    # Plot SF
    for b in data_blocks:
        if b['type'] == 'SF':
            off = result.params[b['offset_key']].value
            ax1.plot(b['t_raw'], b['y_raw'], 'o', color='lightgray', alpha=0.3, markersize=2)
            ax1.plot(b['t'], b['y'], 'kx', alpha=0.5, markersize=3)
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            ax1.plot(t_sm, hybrid_model(t_sm, sf_amps, rates, t_half_D, off), 'r--', lw=1.5)

    ax1.set_title("SF Data (Sigmoid Phase D)")
    ax1.set_ylabel("Voltage (V)")
    # ax1.set_xscale('log') # Optional

    # Plot CD
    cd_amps = [result.params[f'amp_cd_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    off_cd = result.params['off_cd'].value
    for b in data_blocks:
        if b['type'] == 'CD':
            ax2.plot(b['t_raw'] / 3600, b['y_raw'], 'bo', alpha=0.4)
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            ax2.plot(t_sm / 3600, hybrid_model(t_sm, cd_amps, rates, t_half_D, off_cd), 'k-', lw=2)

    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Ellipticity")

    plt.tight_layout()
    plt.show()