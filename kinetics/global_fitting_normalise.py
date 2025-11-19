import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from lmfit import Minimizer, Parameters, report_fit
import re


# ==========================================
# 1. ROBUST LOADERS
# ==========================================

def extract_section(lines, section_name):
    start = -1
    for i, line in enumerate(lines):
        if section_name in line:
            start = i + 2
            break
    if start == -1: return None
    section_lines = []
    for line in lines[start:]:
        if not line.strip(): break
        section_lines.append(line)
    return section_lines


def load_complex_cd_spectra(filename, target_wl=222, dead_time=400):
    if not filename or not os.path.exists(filename): return None, None
    print(f"Processing CD Spectra (DT={dead_time}s): {os.path.basename(filename)}")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        cd_lines = extract_section(lines, "CircularDichroism")
        if not cd_lines: return None, None

        cd_df = pd.read_csv(StringIO(''.join(cd_lines)), skipinitialspace=True)
        cd_df.rename(columns={cd_df.columns[0]: "Time"}, inplace=True)
        cd_df.dropna(axis=1, how='all', inplace=True)

        times = pd.to_numeric(cd_df['Time'], errors='coerce').values
        wl_cols = [c for c in cd_df.columns if c != 'Time']
        wl_vals = pd.to_numeric(wl_cols, errors='coerce')

        valid_wl_mask = ~np.isnan(wl_vals)
        if not np.any(valid_wl_mask): return None, None

        actual_wls = wl_vals[valid_wl_mask]
        actual_cols = np.array(wl_cols)[valid_wl_mask]

        idx = np.argmin(np.abs(actual_wls - target_wl))
        target_col = actual_cols[idx]

        print(f"  Target: {target_wl} nm -> Found: {actual_wls[idx]} nm")

        intensities = pd.to_numeric(cd_df[target_col], errors='coerce').values
        mask = ~np.isnan(times) & ~np.isnan(intensities)
        return times[mask] + dead_time, intensities[mask]

    except Exception as e:
        print(f"  Error parsing CD Spectra: {e}");
        return None, None


def read_cd_kinetics_simple(filename, dead_time=30):
    if not filename or not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            if not re.match(r'^[\d\.\-]', f.readline().strip()):
                skiprows = 1
            else:
                skiprows = 0
        df = pd.read_csv(filename, skiprows=skiprows, header=None).dropna(how='all', axis=1)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]
        t, idx = np.unique(t, return_index=True)
        print(f"Loaded CD Kinetics (DT={dead_time}s): {os.path.basename(filename)}")
        return t + dead_time, y[idx]
    except:
        return None, None


def read_sf_data(filename):
    if not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            skiprows = 1
            if not re.match(r'^[\d\.\-]', f.readline().strip()):
                0
            else:
                0
        df = pd.read_csv(filename, header=None, usecols=[0, 1]).dropna()
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]
        wrap = np.where(np.diff(t) < 0)[0]
        if len(wrap) > 0: t, y = t[:wrap[0] + 1], y[:wrap[0] + 1]
        return t, y
    except:
        return None, None


def get_files(folder, protein):
    if not os.path.exists(folder): return []
    search = 'dzeta' if protein == 'zeta' else protein
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv") and search in f]


def cut_decimate(t, y, t_min=None, t_max=None, max_pts=300):
    if t is None: return None, None
    mask = (t >= (t_min if t_min else -999999)) & (t <= (t_max if t_max else 999999))
    t, y = t[mask], y[mask]
    if len(t) == 0: return None, None
    step = int(np.ceil(len(t) / max_pts))
    return t[::step], y[::step]


# ==========================================
# 2. FRACTIONAL MODEL
# ==========================================

def fractional_model(t, fractions, rates, total_signal, offset):
    kinetics = np.zeros_like(t, dtype=float)
    for frac, k in zip(fractions, rates):
        exponent = np.clip(-k * t, -700, 700)
        kinetics += frac * np.exp(exponent)
    return offset + (total_signal * kinetics)


def objective(params, data_blocks):
    residuals = []
    rates = [params[f'k_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    fractions = [params[f'frac_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]

    for block in data_blocks:
        t, y = block['t'], block['y']
        total_signal = params[block['span_key']]
        offset = params[block['offset_key']]

        model = fractional_model(t, fractions, rates, total_signal, offset)
        weight = 3.0 if block['type'] == 'CD' else 1.0
        residuals.append((y - model) * weight)

    return np.concatenate(residuals)


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    PROTEIN = "gamma"
    TARGET_WL = 222

    # PATHS
    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"
    paths = {
        'SF_Fast': os.path.join(base_sf, "phase_A"),
        'SF_Slow': os.path.join(base_sf, "double_exp_B-C")
    }
    cd_spectra = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"
    cd_kinetics = f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_{PROTEIN}"

    CUT_CONFIG = {
        'SF_Fast': {'min': 0.002, 'max': 1.0},
        'SF_Slow': {'min': 1.0, 'max': None},
        'CD': {'min': None, 'max': None}
    }

    data_blocks = []
    params = Parameters()

    # Rates
    params.add('k_A', value=np.log(2) / 0.38, min=0)
    params.add('k_B', value=np.log(2) / 12.0, min=0)
    params.add('k_C', value=np.log(2) / 334.0, min=0)
    params.add('k_D', value=np.log(2) / 1354.0, min=0)
    params.add('k_E', value=np.log(2) / (2.4 * 3600), min=0)
    params.add('k_F', value=np.log(2) / (33 * 3600), min=0)

    # Fractions
    params.add('frac_A', value=0.4, min=0, max=1)
    params.add('frac_B', value=0.2, min=0, max=1)
    params.add('frac_C', value=0.1, min=0, max=1)
    params.add('frac_D', value=0.05, min=0, max=1)
    params.add('frac_E', value=0.0, vary=False)
    params.add('frac_F', value=0.0, vary=False)

    # Spans
    params.add('span_SF', value=5.0)
    params.add('span_CD', value=200.0)

    # Load SF
    for cat, folder in paths.items():
        files = get_files(folder, PROTEIN)
        for i, fpath in enumerate(files):
            t, y = read_sf_data(fpath)
            if t is not None:
                lim = CUT_CONFIG.get(cat, {})
                t, y = cut_decimate(t, y, lim.get('min'), lim.get('max'))
                if t is not None:
                    off_key = f"off_{cat}_{i}"
                    params.add(off_key, value=y[-1])
                    data_blocks.append({'t': t, 'y': y, 'type': 'SF', 'span_key': 'span_SF', 'offset_key': off_key,
                                        'name': f"{cat}-{i}"})

    # Load CD
    cd_sources = []
    t_spec, y_spec = load_complex_cd_spectra(cd_spectra, target_wl=TARGET_WL, dead_time=400)
    if t_spec is not None: cd_sources.append(('CD_Spectra', t_spec, y_spec))

    f_kin = get_files(cd_kinetics, PROTEIN)[0] if os.path.exists(cd_kinetics) else None
    if f_kin:
        t_kin, y_kin = read_cd_kinetics_simple(f_kin, dead_time=30)
        if t_kin is not None: cd_sources.append(('CD_Kinetics', t_kin, y_kin))

    for name, t, y in cd_sources:
        t, y = cut_decimate(t, y)
        if t is not None:
            off_key = f"off_{name}"
            params.add(off_key, value=y[-1])
            data_blocks.append(
                {'t': t, 'y': y, 'type': 'CD', 'span_key': 'span_CD', 'offset_key': off_key, 'name': name})

    if not data_blocks: print("No data."); exit()

    # Fit
    print(f"\nRunning Fractional Global Fit on {len(data_blocks)} datasets...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))
    result = minner.minimize(method='leastsq')

    # Report
    print("\n" + "=" * 30 + "\n FRACTIONAL FIT RESULTS\n" + "=" * 30)
    for p in ['A', 'B', 'C', 'D', 'E', 'F']:
        k = result.params[f'k_{p}'].value
        if k > 1e-10:
            val = np.log(2) / k;
            unit = "s"
            if val > 3600: val /= 3600; unit = "h"
            print(f"  Phase {p}: {val:.2f} {unit}")

    # --- PLOT NORMALIZED (LINEAR SCALE) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    fractions = [result.params[f'frac_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    rates = [result.params[f'k_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]

    # Determine max time for plot limit
    max_time = 0
    for b in data_blocks:
        if np.max(b['t']) > max_time: max_time = np.max(b['t'])

    # Linear smooth line
    t_smooth = np.linspace(0, max_time, 2000)
    y_smooth_norm = np.zeros_like(t_smooth)
    for f, k in zip(fractions, rates):
        y_smooth_norm += f * np.exp(-k * t_smooth)

    # Plot Points
    for b in data_blocks:
        total_signal = result.params[b['span_key']].value
        offset = result.params[b['offset_key']].value
        y_norm = (b['y'] - offset) / total_signal

        color = 'red' if 'SF' in b['type'] else 'blue'
        marker = '.' if 'SF' in b['type'] else 'o'
        lbl = b['name'] if '0' in b['name'] or 'Spectra' in b['name'] else ""
        ax.plot(b['t'], y_norm, marker, color=color, alpha=0.4, label=lbl)

    # Plot Fit
    ax.plot(t_smooth, y_smooth_norm, 'k-', linewidth=2, label='Global Model')

    # Use Linear Scale
    ax.set_xscale('linear')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude (0 to 1)")
    ax.set_title(f"Global Fit: Normalized Overlay ({PROTEIN}) - Linear Scale")
    ax.legend()
    plt.tight_layout()
    plt.show()