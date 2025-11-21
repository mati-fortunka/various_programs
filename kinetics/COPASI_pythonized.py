import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.integrate import odeint
from lmfit import Minimizer, Parameters, report_fit


# ==========================================
# 1. ROBUST LOADERS (Customized for your files)
# ==========================================

def load_complex_cd_spectra(filename, target_wl=222, dead_time=400):
    """
    Parses ProDataCSV. Extracts ONLY the CircularDichroism block.
    Stops reading when it hits 'HV', 'Voltage', or empty lines.
    """
    if not filename or not os.path.exists(filename):
        print(f"  Warning: Spectra file not found: {filename}")
        return None, None

    print(f"Processing CD Spectra (DT={dead_time}s): {os.path.basename(filename)}")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 1. Locate Start of CD Section
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                start_idx = i
                break

        if start_idx == -1:
            print("  Error: 'CircularDichroism' section not found.")
            return None, None

        # 2. Locate Header Row (Starts with comma: ",250,249.5...")
        # It is usually 2 lines below "CircularDichroism" (skipping "Repeat,Wavelength")
        header_idx = -1
        for j in range(1, 10):
            if start_idx + j >= len(lines): break
            line = lines[start_idx + j].strip()
            # Look for the line listing wavelengths
            if line.startswith(',') and any(c.isdigit() for c in line):
                header_idx = start_idx + j
                break

        if header_idx == -1:
            print("  Error: Wavelength header row not found.")
            return None, None

        # 3. Extract ONLY Data Lines (Stop at next section)
        data_lines = [lines[header_idx]]  # Start with header
        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped: break  # Stop at empty line
            if stripped in ["HV", "Voltage", "Temperature"]: break  # Stop at next section
            # Ensure line starts with a number (Time)
            if re.match(r'^\d', stripped):
                data_lines.append(line)

        # 4. Parse to DataFrame
        # The header starts with ",", so pandas sees the first col as "Unnamed: 0" (Time)
        df = pd.read_csv(StringIO(''.join(data_lines)))

        # 5. Find Target Wavelength Column
        # Columns 1 onwards are wavelengths
        wl_cols = pd.to_numeric(df.columns[1:], errors='coerce')
        valid_mask = ~np.isnan(wl_cols)

        if not np.any(valid_mask): return None, None

        wls = wl_cols[valid_mask]
        idx = np.argmin(np.abs(wls - target_wl))

        # Map back to dataframe column name
        # valid_mask indices map to df.columns[1:]
        # We need the actual column name from df
        target_col = df.columns[1:][valid_mask][idx]
        found_wl = wls[idx]

        print(f"  -> Found WL {found_wl} nm (Column: {target_col})")

        # 6. Extract Time and Intensity
        times = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        intensities = pd.to_numeric(df[target_col], errors='coerce').values

        mask = ~np.isnan(times) & ~np.isnan(intensities)
        times, intensities = times[mask], intensities[mask]

        # Sort
        p = np.argsort(times)
        return times[p] + dead_time, intensities[p]

    except Exception as e:
        print(f"  Error parsing Spectra: {e}");
        return None, None


def read_cd_kinetics_simple(filename, dead_time=30):
    """
    Reads simple kinetics. CUTS data when time resets (to avoid reading HV/Voltage blocks).
    """
    if not filename or not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            # Check if header exists (non-digit first char)
            skip = 1 if not re.match(r'^[\d\.\-]', f.readline().strip()) else 0

        df = pd.read_csv(filename, skiprows=skip, header=None).dropna(how='all', axis=1)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values

        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]

        if len(t) == 0: return None, None

        # --- CRITICAL: DETECT TIME RESET ---
        # Finds where t goes from 2000 -> 10
        diffs = np.diff(t)
        reset_indices = np.where(diffs < -1)[0]  # Big negative jump

        if len(reset_indices) > 0:
            cutoff = reset_indices[0] + 1
            # print(f"  -> Cutting simple kinetics at index {cutoff} (Time reset)")
            t, y = t[:cutoff], y[:cutoff]

        return t + dead_time, y
    except:
        return None, None


def read_sf_data(filename):
    """Reads SF data, cuts on time reset."""
    if not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            skip = 1 if not re.match(r'^[\d\.\-]', f.readline().strip()) else 0
        df = pd.read_csv(filename, header=None, usecols=[0, 1]).dropna()
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values

        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]

        # Time reset check
        diffs = np.diff(t)
        reset_indices = np.where(diffs < 0)[0]
        if len(reset_indices) > 0:
            t, y = t[:reset_indices[0] + 1], y[:reset_indices[0] + 1]

        return t, y
    except:
        return None, None


def get_all_files(folder, protein):
    if not os.path.exists(folder): return []
    s = 'dzeta' if protein == 'zeta' else protein
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv") and s in f]


def cut_data(t, y, settings_key):
    if t is None: return None, None
    cfg = CUT_CONFIG.get(settings_key, {})
    t_min = cfg.get('min', -1e9)
    t_max = cfg.get('max', 1e9)
    # Handle None values in config
    _min = t_min if t_min is not None else -1e9
    _max = t_max if t_max is not None else 1e9
    mask = (t >= _min) & (t <= _max)
    return t[mask], y[mask]


def decimate(t, y, max_pts=1000):
    if t is None or len(t) <= max_pts: return t, y
    step = int(np.ceil(len(t) / max_pts))
    return t[::step], y[::step]


# ==========================================
# 2. ODE MECHANISM
# ==========================================

def kinetic_mechanism(y, t, k_rates):
    N, I1, I2, I3, I4, I5, U = y
    kA, kB, kC, kD, kE, kF = k_rates
    dN = -kA * N
    dI1 = kA * N - kB * I1
    dI2 = kB * I1 - kC * I2
    dI3 = kC * I2 - kD * I3
    dI4 = kD * I3 - kE * I4
    dI5 = kE * I4 - kF * I5
    dU = kF * I5
    return [dN, dI1, dI2, dI3, dI4, dI5, dU]


def solve_mechanism(t, k_params, spec_factors, offset, scale=1.0):
    k_rates = [k_params[f'k_{p}'] for p in ['A', 'B', 'C', 'D', 'E', 'F']]
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if len(t) == 0: return np.array([])

    if t.min() > 0:
        t_sim = np.concatenate(([0.0], t))
        trim = True
    else:
        t_sim = t
        trim = False

    try:
        sort_idx = np.argsort(t_sim)
        t_sorted = t_sim[sort_idx]
        t_unique, u_idx = np.unique(t_sorted, return_index=True)

        conc_matrix_unique = odeint(kinetic_mechanism, y0, t_unique, args=(k_rates,))

        if len(t_unique) != len(t_sorted):
            conc_matrix_sorted = np.zeros((len(t_sorted), 7))
            for i in range(7):
                conc_matrix_sorted[:, i] = np.interp(t_sorted, t_unique, conc_matrix_unique[:, i])
        else:
            conc_matrix_sorted = conc_matrix_unique

        conc_matrix = np.zeros_like(conc_matrix_sorted)
        conc_matrix[sort_idx] = conc_matrix_sorted

        if trim: conc_matrix = conc_matrix[1:]
    except:
        return np.ones_like(t) * 1e6

    factors = np.array(spec_factors)
    signal_curve = np.dot(conc_matrix, factors)
    return offset + (scale * signal_curve)


def objective(params, data_blocks):
    residuals = []
    k_names = ['k_A', 'k_B', 'k_C', 'k_D', 'k_E', 'k_F']
    k_vals = {k: params[k].value for k in k_names}

    for block in data_blocks:
        t, y = block['t'], block['y']
        offset = params[block['offset_key']].value
        scale = params[block['scale_key']].value

        if block['type'] == 'SF':
            specs = [params[f'spec_sf_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'I5', 'U']]
        else:
            specs = [params[f'spec_cd_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'I5', 'U']]

        model = solve_mechanism(t, k_vals, specs, offset, scale)
        if len(model) != len(y): continue

        weight = 3.0 if block['type'] == 'CD' else 1.0
        residuals.append((y - model) * weight)

    return np.concatenate(residuals) if residuals else np.array([1e6])


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    PROTEIN = "gamma"
    TARGET_WL = 222

    DT_KINETICS = 30
    DT_SPECTRA = 400

    # CUTS
    CUT_CONFIG = {
        'SF_Fast': {'min': 0.1, 'max': None},
        'SF_Slow': {'min': 1.0, 'max': None},
        'CD': {'min': None, 'max': None}
    }

    # PATHS
    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"
    paths = {'SF_Fast': os.path.join(base_sf, "phase_A"), 'SF_Slow': os.path.join(base_sf, "double_exp_B-C")}
    cd_spectra_file = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"
    cd_kinetics_folder = f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_{PROTEIN}"

    data_blocks = []
    params = Parameters()

    # 1. SF Load
    for cat, folder in paths.items():
        for i, fpath in enumerate(get_all_files(folder, PROTEIN)):
            t_raw, y_raw = read_sf_data(fpath)
            t_cut, y_cut = cut_data(t_raw, y_raw, cat)
            if t_cut is not None and len(t_cut) > 10:
                t_dec, y_dec = decimate(t_cut, y_cut, 500)
                off_k, scale_k = f"off_sf_{cat}_{i}", f"scale_sf_{cat}_{i}"
                params.add(off_k, value=0);
                params.add(scale_k, value=1.0, min=0.5, max=1.5)
                data_blocks.append({'t': t_dec, 'y': y_dec, 'type': 'SF', 'name': f"{cat}-{i}", 'offset_key': off_k,
                                    'scale_key': scale_k})

    # 2. CD Spectra Load
    t_spec, y_spec = load_complex_cd_spectra(cd_spectra_file, TARGET_WL, DT_SPECTRA)
    if t_spec is not None and len(t_spec) > 5:
        t_dec, y_dec = decimate(t_spec, y_spec, 1000)
        params.add('off_cd_spec', value=0)
        params.add('scale_cd_spec', value=1.0, min=0.1, max=2.0)
        data_blocks.append({'t': t_dec, 'y': y_dec, 'type': 'CD', 'name': 'Spectra', 'offset_key': 'off_cd_spec',
                            'scale_key': 'scale_cd_spec'})

    # 3. CD Kinetics Load
    for i, fpath in enumerate(get_all_files(cd_kinetics_folder, PROTEIN)):
        t_kin, y_kin = read_cd_kinetics_simple(fpath, DT_KINETICS)
        if t_kin is not None and len(t_kin) > 5:
            t_dec, y_dec = decimate(t_kin, y_kin, 500)
            off_k, scale_k = f"off_cd_k{i}", f"scale_cd_k{i}"
            params.add(off_k, value=0);
            params.add(scale_k, value=1.0, min=0.1, max=2.0)
            data_blocks.append(
                {'t': t_dec, 'y': y_dec, 'type': 'CD', 'name': f'Kin-{i}', 'offset_key': off_k, 'scale_key': scale_k})

    if not data_blocks: print("CRITICAL: No Data Loaded"); exit()

    # 4. Params
    params.add('k_A', value=np.log(2) / 0.38, min=0)
    params.add('k_B', value=np.log(2) / 12.0, min=0)
    params.add('k_C', value=np.log(2) / 334.0, min=0)
    params.add('k_D', value=np.log(2) / 1354.0, min=0)
    params.add('k_E', value=np.log(2) / (2.4 * 3600), min=0)
    params.add('k_F', value=np.log(2) / (33 * 3600), min=0)

    # CD Factors (mdeg) - Based on your Anchors (N=-14.6, U=0)
    # We assume the signal effectively goes UP from -14 to 0.

    params.add('spec_cd_N', value=-14.6, vary=False)  # Fixed Start
    params.add('spec_cd_I1', value=-12.0)  # Slight loss of structure
    params.add('spec_cd_I2', value=-10.0)
    params.add('spec_cd_I3', value=-8.0)
    params.add('spec_cd_I4', value=-5.0)  # Major loss of structure
    params.add('spec_cd_I5', value=-2.0)
    params.add('spec_cd_U', value=0.0, vary=False)  # Fixed End

    # SF factors (start high, end low)
    for s in ['N', 'I1', 'I2', 'I3', 'I4', 'I5', 'U']:
        params.add(f'spec_sf_{s}', value=10.0)

    # 5. Fit
    print(f"\nRunning ODE Fit on {len(data_blocks)} datasets...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))
    result = minner.minimize(method='leastsq')

    # 6. Report
    print("\n" + "=" * 50 + "\n RESULTS: ODE MECHANISM\n" + "=" * 50)
    four_hours = 14400
    for p in ['A', 'B', 'C', 'D', 'E', 'F']:
        k = result.params[f'k_{p}'].value
        k_err = result.params[f'k_{p}'].stderr or 0
        if k > 1e-12:
            th = np.log(2) / k
            th_err = (np.log(2) / k ** 2) * k_err
            if th <= four_hours:
                print(f"Step {p}: {th:.2f} ± {th_err:.2f} s")
            else:
                print(f"Step {p}: {th / 3600:.2f} ± {th_err / 3600:.2f} h")

    # 7. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    k_final = {k: result.params[k].value for k in ['k_A', 'k_B', 'k_C', 'k_D', 'k_E', 'k_F']}
    spec_sf = [result.params[f'spec_sf_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'I5', 'U']]
    spec_cd = [result.params[f'spec_cd_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'I5', 'U']]

    # SF Plot (Linear)
    for b in data_blocks:
        if b['type'] == 'SF':
            off, scale = result.params[b['offset_key']].value, result.params[b['scale_key']].value
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            model = solve_mechanism(t_sm, k_final, spec_sf, off, scale)
            ax1.plot(b['t'], b['y'], 'o', color='lightgray', ms=2)
            ax1.plot(t_sm, model, 'r-', lw=1)
    ax1.set_xlabel("Time (s)");
    ax1.set_ylabel("SF Signal (V)")
    ax1.set_title("Mechanistic Fit: SF Data")

    # CD Plot (Linear with Inset)
    ax_ins = ax2.inset_axes([0.45, 0.08, 0.50, 0.35])
    max_t = max([max(b['t']) for b in data_blocks if b['type'] == 'CD'])

    for b in data_blocks:
        if b['type'] == 'CD':
            off, scale = result.params[b['offset_key']].value, result.params[b['scale_key']].value
            col = 'blue' if 'Spectra' in b['name'] else 'green'
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            model = solve_mechanism(t_sm, k_final, spec_cd, off, scale)

            ax2.plot(b['t'] / 3600, b['y'], 'o', color=col, ms=3, alpha=0.3)
            ax2.plot(t_sm / 3600, model, 'k--', lw=1.5)

            ax_ins.plot(b['t'] / 3600, b['y'], 'o', color=col, ms=4, alpha=0.4)
            ax_ins.plot(t_sm / 3600, model, 'k--', lw=1.5)

    ax2.set_xlim(0, max_t / 3600 * 1.05);
    ax2.set_xlabel("Time (h)");
    ax2.set_ylabel("Ellipticity")
    ax_ins.set_xlim(-0.01, 2000 / 3600);
    ax_ins.set_title("Zoom 0-2000s")
    ax2.indicate_inset_zoom(ax_ins, edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/ODE_fits/lin_model/ODE_fit9.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()