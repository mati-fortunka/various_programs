import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from lmfit import Minimizer, Parameters, report_fit


# ==========================================
# 1. ROBUST LOADERS
# ==========================================

def extract_section(lines, section_name):
    start = -1
    for i, line in enumerate(lines):
        if section_name in line:
            start = i + 2  # Data starts 2 lines after section header usually
            break
    if start == -1: return None
    section_lines = []
    for line in lines[start:]:
        if not line.strip(): break
        section_lines.append(line)
    return section_lines


def load_complex_cd_spectra(filename, target_wl=222, dead_time=400):
    """
    Parses ProDataCSV format where Wavelengths are columns and Time is rows.
    """
    if not filename or not os.path.exists(filename):
        print(f"  Error: File not found: {filename}")
        return None, None

    print(f"Processing CD Spectra (DT={dead_time}s): {os.path.basename(filename)}")

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Locate "CircularDichroism" section
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                start_idx = i + 2  # Skip "Repeat,Wavelength" line
                break

        if start_idx == -1:
            print("  Error: 'CircularDichroism' section not found.")
            return None, None

        # The line at start_idx contains wavelengths: ",250,249.5,..."
        # We need to manually parse this because read_csv might get confused by the leading comma
        header_line = lines[start_idx].strip()
        if header_line.startswith(','):
            header_line = "Time" + header_line  # Give the first column a name

        # Combine header with data lines
        data_lines = [header_line]
        for line in lines[start_idx + 1:]:
            if not line.strip(): break  # Stop at empty line
            data_lines.append(line)

        # Create DataFrame
        df = pd.read_csv(StringIO('\n'.join(data_lines)))

        # 1. Identify Wavelength Columns
        # Convert columns to numeric, coercing errors (like 'Time') to NaN
        wl_cols = pd.to_numeric(df.columns, errors='coerce')
        valid_cols = ~np.isnan(wl_cols)

        if not np.any(valid_cols):
            print("  Error: No numeric wavelength columns found.")
            return None, None

        actual_wls = wl_cols[valid_cols]
        actual_col_names = df.columns[valid_cols]

        # 2. Find Closest Wavelength
        idx = np.argmin(np.abs(actual_wls - target_wl))
        target_col_name = actual_col_names[idx]
        found_wl = actual_wls[idx]
        print(f"  -> Target: {target_wl} nm, Found: {found_wl} nm")

        # 3. Extract Data
        # Time is the first column
        times = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        intensities = pd.to_numeric(df[target_col_name], errors='coerce').values

        # Clean
        mask = ~np.isnan(times) & ~np.isnan(intensities)
        times = times[mask]
        intensities = intensities[mask]

        # Sort
        sort_idx = np.argsort(times)

        print(f"  -> Loaded {len(times)} points. Time range: {times[sort_idx][0]} - {times[sort_idx][-1]}")

        return times[sort_idx] + dead_time, intensities[sort_idx]

    except Exception as e:
        print(f"  Error parsing CD Spectra: {e}")
        return None, None


# ... [Rest of loaders: read_cd_kinetics_simple, read_sf_data, etc. remain the same] ...
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


def get_all_files(folder, protein):
    if not os.path.exists(folder): return []
    search = 'dzeta' if protein == 'zeta' else protein
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv") and search in f]


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

def hybrid_model_base(t, amps, rates, t_half_D):
    """Base model (Pure Kinetics centered at 0)"""
    model = np.zeros_like(t, dtype=float)
    exp_indices = [0, 1, 2, 4, 5]
    for i in exp_indices:
        exponent = np.clip(-rates[i] * t, -700, 700)
        model += amps[i] * np.exp(exponent)
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
        else:
            amps = [params[f'amp_cd_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]

        offset = params[block['offset_key']]
        scale = params[block['scale_key']]

        base_signal = hybrid_model_base(t, amps, rates, t_half_D)
        model = offset + (scale * base_signal)

        weight = 5.0 if block['type'] == 'CD' else 1.0
        residuals.append((y - model) * weight)

    return np.concatenate(residuals)


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":

    # --- USER SETTINGS ---
    PROTEIN = "gamma"
    TARGET_WL = 222
    DT_KINETICS = 30
    DT_SPECTRA = 400

    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"
    paths = {
        'SF_Fast': os.path.join(base_sf, "phase_A"),
        'SF_Slow': os.path.join(base_sf, "double_exp_B-C")
    }
    cd_spectra_file = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"
    cd_kinetics_folder = f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_{PROTEIN}"

    CUT_CONFIG = {
        'SF_Fast': {'min': 0.002, 'max': None},
        'SF_Slow': {'min': 1.0, 'max': None},
        'CD': {'min': None, 'max': None}
    }

    data_blocks = []
    params = Parameters()

    # 1. SF
    for category, folder in paths.items():
        files = get_all_files(folder, PROTEIN)
        print(f"Found {len(files)} files in {category}")
        for i, fpath in enumerate(files):
            t_raw, y_raw = read_sf_data(fpath)
            if t_raw is not None:
                lim = CUT_CONFIG.get(category, {})
                t_cut, y_cut = cut_data(t_raw, y_raw, lim.get('min'), lim.get('max'))
                if t_cut is not None and len(t_cut) > 10:
                    t_dec, y_dec = decimate(t_cut, y_cut, 500)
                    off_key = f"off_sf_{category}_{i}"
                    scale_key = f"scale_sf_{category}_{i}"
                    params.add(off_key, value=np.mean(y_dec))
                    params.add(scale_key, value=1.0, min=0.5, max=1.5)
                    data_blocks.append({
                        't': t_dec, 'y': y_dec, 't_raw': t_cut, 'y_raw': y_cut,
                        'type': 'SF', 'name': f"{category}-{i}", 'offset_key': off_key, 'scale_key': scale_key
                    })

    # 2. CD SPECTRA (60h)
    t_spec, y_spec = load_complex_cd_spectra(cd_spectra_file, target_wl=TARGET_WL, dead_time=DT_SPECTRA)
    if t_spec is not None:
        t_dec, y_dec = decimate(t_spec, y_spec, 1000)
        params.add('off_cd_spec', value=y_dec[-1])
        params.add('scale_cd_spec', value=1.0, min=0.1, max=2.0)
        data_blocks.append({
            't': t_dec, 'y': y_dec, 't_raw': t_spec, 'y_raw': y_spec,
            'type': 'CD', 'name': 'CD-Spectra', 'offset_key': 'off_cd_spec', 'scale_key': 'scale_cd_spec'
        })
    else:
        print("  WARNING: Spectra file loading failed. Check format.")

    # 3. CD KINETICS (Short)
    kin_files = get_all_files(cd_kinetics_folder, PROTEIN)
    print(f"Found {len(kin_files)} simple CD kinetic files.")
    for i, fpath in enumerate(kin_files):
        t_kin, y_kin = read_cd_kinetics_simple(fpath, dead_time=DT_KINETICS)
        if t_kin is not None:
            t_dec, y_dec = decimate(t_kin, y_kin, 500)
            off_key = f'off_cd_kin_{i}'
            scale_key = f'scale_cd_kin_{i}'
            params.add(off_key, value=y_dec[-1])
            params.add(scale_key, value=1.0, min=0.1, max=2.0)
            data_blocks.append({
                't': t_dec, 'y': y_dec, 't_raw': t_kin, 'y_raw': y_kin,
                'type': 'CD', 'name': f'CD-Kin-{i}', 'offset_key': off_key, 'scale_key': scale_key
            })

    if not data_blocks: print("No data loaded."); exit()

    # PARAMS
    params.add('k_A', value=np.log(2) / 0.38, min=0)
    params.add('k_B', value=np.log(2) / 12.0, min=0)
    params.add('k_C', value=np.log(2) / 334.0, min=0)
    params.add('k_D', value=0.01, min=0)
    params.add('t_half_D', value=1500.0, min=1000, max=5000)
    params.add('k_E', value=np.log(2) / (2.4 * 3600), min=0)
    params.add('k_F', value=np.log(2) / (33 * 3600), min=0)

    params.add('amp_sf_A', value=2.0);
    params.add('amp_sf_B', value=1.0)
    params.add('amp_sf_C', value=0.5);
    params.add('amp_sf_D', value=0.2)
    params.add('amp_sf_E', value=0, vary=False);
    params.add('amp_sf_F', value=0, vary=False)

    params.add('amp_cd_A', value=0, vary=False);
    params.add('amp_cd_B', value=0)
    params.add('amp_cd_C', value=500);
    params.add('amp_cd_D', value=0)
    params.add('amp_cd_E', value=2000);
    params.add('amp_cd_F', value=5000)

    # FIT
    print(f"\nRunning Hybrid Global Fit ({len(data_blocks)} blocks)...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))
    result = minner.minimize(method='leastsq')

    # REPORT
    print("\n" + "=" * 60 + "\n RESULTS: HYBRID MODEL\n" + "=" * 60)
    four_hours = 14400
    for p in ['A', 'B', 'C', 'E', 'F']:
        k = result.params[f'k_{p}'].value
        k_err = result.params[f'k_{p}'].stderr
        if k > 1e-10:
            th = np.log(2) / k
            th_err = (np.log(2) / (k ** 2)) * k_err if k_err else 0.0
            if th <= four_hours:
                print(f"Phase {p} (Exp): {th:.2f} ± {th_err:.2f} s  ({th / 3600:.3f} h)")
            else:
                print(f"Phase {p} (Exp): {th / 3600:.2f} ± {th_err / 3600:.2f} h")
    th_D = result.params['t_half_D'].value
    th_D_err = result.params['t_half_D'].stderr or 0.0
    slope_D = result.params['k_D'].value
    if th_D <= four_hours:
        print(f"Phase D (Sigm): {th_D:.2f} ± {th_D_err:.2f} s ({th_D / 3600:.3f} h)")
    else:
        print(f"Phase D (Sigm): {th_D / 3600:.2f} ± {th_D_err / 3600:.2f} h")

    # PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    rates = [result.params[f'k_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    t_half_D = result.params['t_half_D'].value

    # 1: SF
    sf_amps = [result.params[f'amp_sf_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    for b in data_blocks:
        if b['type'] == 'SF':
            off = result.params[b['offset_key']].value
            scale = result.params[b['scale_key']].value
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            model_sm = off + (scale * hybrid_model_base(t_sm, sf_amps, rates, t_half_D))
            ax1.plot(b['t_raw'], b['y_raw'], 'o', color='lightgray', alpha=0.3, markersize=2)
            ax1.plot(t_sm, model_sm, 'r--', lw=1.0)
    ax1.set_ylabel("SF Signal (V)");
    ax1.set_title("SF Data")

    # 2: CD
    cd_amps = [result.params[f'amp_cd_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    ax_ins = ax2.inset_axes([0.45, 0.08, 0.50, 0.35])

    all_cd_times = [t for b in data_blocks if b['type'] == 'CD' for t in b['t']]
    max_cd_time = max(all_cd_times) if all_cd_times else 60 * 3600

    for b in data_blocks:
        if b['type'] == 'CD':
            off = result.params[b['offset_key']].value
            scale = result.params[b['scale_key']].value
            col = 'blue' if 'Spectra' in b['name'] else 'green'
            lbl = 'Spectra' if 'Spectra' in b['name'] else 'Kinetics'

            # Plot Main
            ax2.plot(b['t_raw'] / 3600, b['y_raw'], 'o', color=col, alpha=0.3, markersize=3, label=lbl)
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            model_sm = off + (scale * hybrid_model_base(t_sm, cd_amps, rates, t_half_D))
            ax2.plot(t_sm / 3600, model_sm, 'k--', lw=1.5)

            # Plot Inset
            ax_ins.plot(b['t_raw'] / 3600, b['y_raw'], 'o', color=col, alpha=0.4, markersize=4)
            ax_ins.plot(t_sm / 3600, model_sm, 'k--', lw=1.5)

    # Fix Legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax2.set_xlim(0, max_cd_time / 3600 * 1.05)
    ax2.set_xlabel("Time (h)");
    ax2.set_ylabel("Ellipticity")

    ax_ins.set_xlim(-0.02, 2000 / 3600)
    ax_ins.set_title("Zoom: 0-2000s");
    ax_ins.grid(True, alpha=0.3)
    ax2.indicate_inset_zoom(ax_ins, edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/global fitting/fit6.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()