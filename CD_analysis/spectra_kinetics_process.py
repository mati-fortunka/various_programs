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
    """Parses complex spectra file and adds large dead time."""
    if not filename or not os.path.exists(filename):
        return None, None

    print(f"Processing CD Spectra (DT={dead_time}s): {os.path.basename(filename)}")
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

        # APPLY DEAD TIME
        return times + dead_time, intensities
    except Exception as e:
        print(f"Error parsing CD Spectra: {e}")
        return None, None


def read_cd_kinetics_simple(filename, dead_time=30):
    """Parses simple CD kinetics CSV and adds dead time."""
    if not filename or not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            skiprows = 1 if not re.match(r'^[\d\.\-]', first_line.split(',')[0]) else 0

        df = pd.read_csv(filename, skiprows=skiprows, header=None).dropna(how='all', axis=1)
        # Ensure 2 columns
        if df.shape[1] < 2: return None, None

        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values

        # Filter NaNs
        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]

        # Handle time wrapping
        wrap_idx = np.where(np.diff(t) < 0)[0]
        if len(wrap_idx) > 0:
            t = t[:wrap_idx[0] + 1]
            y = y[:wrap_idx[0] + 1]

        # Remove duplicates
        _, unique_idx = np.unique(t, return_index=True)
        t = t[unique_idx]
        y = y[unique_idx]

        print(f"Loaded CD Kinetics (DT={dead_time}s): {os.path.basename(filename)}")
        return t + dead_time, y

    except Exception as e:
        print(f"Error CD Kinetics: {e}")
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
        return None, None


def get_all_files(folder, protein):
    search = 'dzeta' if protein == 'zeta' else protein
    files = []
    if not os.path.exists(folder): return []
    for f in os.listdir(folder):
        if f.endswith(".csv") and search in f:
            files.append(os.path.join(folder, f))
    return files


def get_single_file(folder, protein):
    """Helper for CD kinetics folder."""
    files = get_all_files(folder, protein)
    return files[0] if files else None


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
# 2. MODEL
# ==========================================

def multi_exp_model(t, amps, rates, offset):
    model = np.full_like(t, offset, dtype=float)
    for A, k in zip(amps, rates):
        exponent = np.clip(-k * t, -700, 700)
        model += A * np.exp(exponent)
    return model


def objective(params, data_blocks):
    residuals = []
    rates = [params[f'k_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]

    for block in data_blocks:
        t, y = block['t'], block['y']

        if block['type'] == 'SF':
            amps = [params[f'amp_sf_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
            offset = params[block['offset_key']]
        else:  # CD
            amps = [params[f'amp_cd_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
            offset = params['off_cd']

        model = multi_exp_model(t, amps, rates, offset)
        weight = 5.0 if block['type'] == 'CD' else 1.0
        residuals.append((y - model) * weight)

    return np.concatenate(residuals)


# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":

    PROTEIN = "gamma"
    TARGET_WL = 222

    # --- DEAD TIMES (Seconds) ---
    DT_KINETICS = 30
    DT_SPECTRA = 400

    # --- PATHS ---
    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"

    # Input Folders/Files
    paths = {
        'SF_Fast': os.path.join(base_sf, "phase_A"),
        'SF_Slow': os.path.join(base_sf, "double_exp_B-C")
    }

    # 1. CD SPECTRA FILE (Complex)
    cd_spectra_file = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"

    # 2. CD KINETICS FOLDER (Simple CSVs) - Set to None if you don't want to use it
    cd_kinetics_folder = f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_{PROTEIN}"

    # --- CUT CONFIG ---
    CUT_CONFIG = {
        'SF_Fast': {'min': 0.002, 'max': 1.0},
        'SF_Slow': {'min': 1.0, 'max': None},
        'CD': {'min': None, 'max': None}
    }

    data_blocks = []
    params = Parameters()

    # --- 1. Load SF Files ---
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

    # --- 2. Load CD Spectra (Complex, DT=400s) ---
    t_spec, y_spec = load_complex_cd_spectra(cd_spectra_file, target_wl=TARGET_WL, dead_time=DT_SPECTRA)
    if t_spec is not None:
        t_dec, y_dec = decimate(t_spec, y_spec, 300)
        data_blocks.append(
            {'t': t_dec, 'y': y_dec, 't_raw': t_spec, 'y_raw': y_spec, 'type': 'CD', 'name': 'CD-Spectra'})
        # Initialize offset with the end of the data
        params.add('off_cd', value=y_dec[-1])

    # --- 3. Load CD Kinetics (Simple, DT=30s) ---
    if cd_kinetics_folder:
        fpath_kin = get_single_file(cd_kinetics_folder, PROTEIN)
        if fpath_kin:
            t_kin, y_kin = read_cd_kinetics_simple(fpath_kin, dead_time=DT_KINETICS)
            if t_kin is not None:
                t_dec, y_dec = decimate(t_kin, y_kin, 300)
                data_blocks.append(
                    {'t': t_dec, 'y': y_dec, 't_raw': t_kin, 'y_raw': y_kin, 'type': 'CD', 'name': 'CD-Kinetics'})
                # If off_cd not set by spectra, set it here
                if 'off_cd' not in params: params.add('off_cd', value=y_dec[-1])

    if not data_blocks: print("No data loaded."); exit()

    # --- Global Params ---
    params.add('k_A', value=np.log(2) / 0.38, min=0)
    params.add('k_B', value=np.log(2) / 12.0, min=0)
    params.add('k_C', value=np.log(2) / 334.0, min=0)
    params.add('k_D', value=np.log(2) / 1354.0, min=0)
    params.add('k_E', value=np.log(2) / (2.4 * 3600), min=0)
    params.add('k_F', value=np.log(2) / (33 * 3600), min=0)

    params.add('amp_sf_A', value=2.0)
    params.add('amp_sf_B', value=1.0)
    params.add('amp_sf_C', value=0.5)
    params.add('amp_sf_D', value=0.2)
    params.add('amp_sf_E', value=0, vary=False)
    params.add('amp_sf_F', value=0, vary=False)

    params.add('amp_cd_A', value=0, vary=False)
    params.add('amp_cd_B', value=100)
    params.add('amp_cd_C', value=500)
    params.add('amp_cd_D', value=1000)
    params.add('amp_cd_E', value=2000)
    params.add('amp_cd_F', value=5000)

    # --- Fit ---
    print(f"\nRunning Global Fit on {len(data_blocks)} datasets...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))
    result = minner.minimize(method='leastsq')

    # --- Report ---
    print("\n" + "=" * 30)
    print(f" RESULTS: {PROTEIN}")
    print("=" * 30)
    for p in ['A', 'B', 'C', 'D', 'E', 'F']:
        k = result.params[f'k_{p}'].value
        err = result.params[f'k_{p}'].stderr
        if k > 1e-10:
            th = np.log(2) / k;
            th_err = (np.log(2) / k ** 2) * err if err else 0
            unit = "s"
            if th > 3600: th /= 3600; th_err /= 3600; unit = "h"
            print(f"Phase {p}: {th:.2f} ± {th_err:.2f} {unit}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    rates = [result.params[f'k_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    sf_amps = [result.params[f'amp_sf_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    cd_amps = [result.params[f'amp_cd_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    off_cd = result.params['off_cd'].value

    # Plot SF
    for b in data_blocks:
        if b['type'] == 'SF':
            off = result.params[b['offset_key']].value
            ax1.plot(b['t_raw'], b['y_raw'], 'o', color='lightgray', alpha=0.3, markersize=2)
            # ax1.plot(b['t'], b['y'], 'kx', alpha=0.5, markersize=3) # Optional: show fit points
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            ax1.plot(t_sm, multi_exp_model(t_sm, sf_amps, rates, off), 'r--', lw=1, alpha=0.8)
    ax1.set_ylabel("SF Signal (V)", fontsize=16);
    # ax1.set_title("Global Fit: SF", fontsize=16)

    # Plot CD
    for b in data_blocks:
        if b['type'] == 'CD':
            color = 'blue' if b['name'] == 'CD-Spectra' else 'green'
            label = b['name']
            ax2.plot(b['t_raw'] / 3600, b['y_raw'], 'o', color=color, alpha=0.3, label=label)

            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            ax2.plot(t_sm / 3600, multi_exp_model(t_sm, cd_amps, rates, off_cd), 'k-', lw=1.5)

    ax2.set_xlabel("Time (h)", fontsize=16);
    ax2.set_ylabel("Ellipticity",fontsize=16);
    # ax2.set_title("Global Fit: CD")
    ax2.legend()
    plt.tight_layout()
    plt.show()