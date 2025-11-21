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
# 1. ROBUST LOADERS
# ==========================================
# (Loaders remain the same, collapsing for brevity)
def extract_section(lines, section_name):
    start = -1
    for i, line in enumerate(lines):
        if section_name in line: start = i + 2; break
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
        start_idx = -1
        for i, line in enumerate(lines):
            if "CircularDichroism" in line: start_idx = i; break
        if start_idx == -1: return None, None
        header_idx = -1
        for j in range(1, 20):
            if start_idx + j >= len(lines): break
            line = lines[start_idx + j].strip()
            if line.startswith(',') and any(c.isdigit() for c in line): header_idx = start_idx + j; break
        if header_idx == -1: return None, None
        data_str = lines[header_idx] + "".join(lines[header_idx + 1:])
        df = pd.read_csv(StringIO(data_str))
        wl_cols = pd.to_numeric(df.columns[1:], errors='coerce')
        valid_mask = ~np.isnan(wl_cols)
        if not np.any(valid_mask): return None, None
        wls = wl_cols[valid_mask]
        idx = np.argmin(np.abs(wls - target_wl))
        target_col_idx = np.where(valid_mask)[0][idx] + 1
        target_col_name = df.columns[target_col_idx]
        print(f"  -> Target: {target_wl} nm, Found: {wls[idx]} nm")
        times = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        intensities = pd.to_numeric(df[target_col_name], errors='coerce').values
        mask = ~np.isnan(times) & ~np.isnan(intensities)
        times, intensities = times[mask], intensities[mask]
        p = np.argsort(times)
        return times[p] + dead_time, intensities[p]
    except:
        return None, None


def read_cd_kinetics_simple(filename, dead_time=30):
    if not filename or not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            skip = 1 if not re.match(r'^[\d\.\-]', f.readline().strip()) else 0
        df = pd.read_csv(filename, skiprows=skip, header=None).dropna(how='all', axis=1)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]
        diffs = np.diff(t)
        resets = np.where(diffs < -1)[0]
        if len(resets) > 0: t, y = t[:resets[0] + 1], y[:resets[0] + 1]
        t, idx = np.unique(t, return_index=True)
        return t + dead_time, y[idx]
    except:
        return None, None


def read_sf_data(filename):
    if not os.path.exists(filename): return None, None
    try:
        with open(filename, 'r') as f:
            skip = 1 if not re.match(r'^[\d\.\-]', f.readline().strip()) else 0
        df = pd.read_csv(filename, header=None, usecols=[0, 1]).dropna()
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[mask], y[mask]
        diffs = np.diff(t)
        resets = np.where(diffs < 0)[0]
        if len(resets) > 0: t, y = t[:resets[0] + 1], y[:resets[0] + 1]
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
    t_min = cfg.get('min', -1e9);
    t_max = cfg.get('max', 1e9)
    _min = t_min if t_min is not None else -1e9
    _max = t_max if t_max is not None else 1e9
    mask = (t >= _min) & (t <= _max)
    return t[mask], y[mask]


def decimate(t, y, max_pts=300):
    if t is None or len(t) <= max_pts: return t, y
    step = int(np.ceil(len(t) / max_pts))
    return t[::step], y[::step]


# ==========================================
# 2. PARALLEL MODEL (ODE)
# ==========================================

def kinetic_mechanism(y, t, k_rates):
    """
    Parallel Paths:
    1. N -> I1 -> I2 -> U (Rates k1, k2, k3)
    2. N -> I3 -> I4 -> U (Rates k4, k5, k6)
    """
    N, I1, I2, I3, I4, U = y
    k1, k2, k3, k4, k5, k6 = k_rates

    # Loss of N into two branches
    dN = -(k1 + k4) * N

    # Branch 1
    dI1 = k1 * N - k2 * I1
    dI2 = k2 * I1 - k3 * I2

    # Branch 2
    dI3 = k4 * N - k5 * I3
    dI4 = k5 * I3 - k6 * I4

    # U formation from both
    dU = k3 * I2 + k6 * I4

    return [dN, dI1, dI2, dI3, dI4, dU]


def solve_mechanism(t, k_params, spec_factors, offset, scale=1.0):
    k_rates = [
        k_params['k1_N_I1'], k_params['k2_I1_I2'], k_params['k3_I2_U'],
        k_params['k4_N_I3'], k_params['k5_I3_I4'], k_params['k6_I4_U']
    ]
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if len(t) == 0: return np.array([])

    # ODE Safety
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

        # INCREASED MXSTEP to prevent crash on stiff systems
        conc_matrix_unique = odeint(kinetic_mechanism, y0, t_unique, args=(k_rates,), mxstep=5000)

        if len(t_unique) != len(t_sorted):
            conc_matrix_sorted = np.zeros((len(t_sorted), 6))
            for i in range(6):
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
    k_names = ['k1_N_I1', 'k2_I1_I2', 'k3_I2_U', 'k4_N_I3', 'k5_I3_I4', 'k6_I4_U']
    k_vals = {k: params[k].value for k in k_names}

    for block in data_blocks:
        t, y = block['t'], block['y']
        offset = params[block['offset_key']].value
        # Note: Scale is removed/fixed to 1.0 to prevent redundancy
        scale = 1.0

        if block['type'] == 'SF':
            specs = [params[f'spec_sf_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'U']]
        else:
            specs = [params[f'spec_cd_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'U']]

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

    base_sf = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics"
    paths = {'SF_Fast': os.path.join(base_sf, "phase_A"), 'SF_Slow': os.path.join(base_sf, "double_exp_B-C")}
    cd_spectra_file = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/G/spectra_kinetics/60h_2/8_3_gamma_spectra_kin_60h00000.csv"
    cd_kinetics_folder = f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_{PROTEIN}"

    CUT_CONFIG = {'SF_Fast': {'min': 0.1, 'max': 1.0}, 'SF_Slow': {'min': 1.0, 'max': None},
                  'CD': {'min': None, 'max': None}}

    data_blocks = []
    params = Parameters()

    # --- COLLECT START/END POINTS FOR ANCHORING ---
    sf_starts, sf_ends = [], []
    cd_starts, cd_ends = [], []

    # 1. LOAD SF
    for cat, folder in paths.items():
        for i, fpath in enumerate(get_all_files(folder, PROTEIN)):
            t_raw, y_raw = read_sf_data(fpath)
            t_cut, y_cut = cut_data(t_raw, y_raw, cat)
            if t_cut is not None and len(t_cut) > 10:
                t_dec, y_dec = decimate(t_cut, y_cut, 500)
                off_k = f"off_sf_{cat}_{i}"
                params.add(off_k, value=0, vary=True)
                data_blocks.append({'t': t_dec, 'y': y_dec, 'type': 'SF', 'name': f"{cat}-{i}", 'offset_key': off_k,
                                    'scale_key': None})
                # Collect stats
                if cat == 'SF_Fast': sf_starts.append(y_dec[0])
                sf_ends.append(y_dec[-1])

    # 2. LOAD SPECTRA
    t_spec, y_spec = load_complex_cd_spectra(cd_spectra_file, TARGET_WL, DT_SPECTRA)
    if t_spec is not None and len(t_spec) > 5:
        t_dec, y_dec = decimate(t_spec, y_spec, 1000)
        params.add('off_cd_spec', value=0)
        data_blocks.append(
            {'t': t_dec, 'y': y_dec, 'type': 'CD', 'name': 'Spectra', 'offset_key': 'off_cd_spec', 'scale_key': None})
        cd_ends.append(y_dec[-1])

    # 3. LOAD KINETICS
    for i, fpath in enumerate(get_all_files(cd_kinetics_folder, PROTEIN)):
        t_kin, y_kin = read_cd_kinetics_simple(fpath, DT_KINETICS)
        if t_kin is not None and len(t_kin) > 5:
            t_dec, y_dec = decimate(t_kin, y_kin, 500)
            off_k = f"off_cd_k{i}"
            params.add(off_k, value=0)
            data_blocks.append(
                {'t': t_dec, 'y': y_dec, 'type': 'CD', 'name': f'Kin-{i}', 'offset_key': off_k, 'scale_key': None})
            cd_starts.append(y_dec[0])

    if not data_blocks: print("CRITICAL: No Data Loaded."); exit()

    # --- CALCULATE ANCHORS ---
    avg_sf_start = np.mean(sf_starts) if sf_starts else 10.0
    avg_sf_end = np.mean(sf_ends) if sf_ends else 10.0  # Likely unfolded
    avg_cd_start = np.mean(cd_starts) if cd_starts else 0.0
    avg_cd_end = np.mean(cd_ends) if cd_ends else -10.0

    print(f"Anchoring SF: N={avg_sf_start:.2f}, U={avg_sf_end:.2f}")
    print(f"Anchoring CD: N={avg_cd_start:.2f}, U={avg_cd_end:.2f}")

    # --- 2. PARAMETERS (CONSTRAINED PARALLEL MODEL) ---

    # We enforce a "Fast -> Medium -> Slow" topology for both paths
    # to prevent the optimizer from making them all 0.0s.

    # --- PATH 1 (Assume this is the SF-dominant fast path) ---
    # Step 1: N -> I1 (Fast: < 1s)
    params.add('k1_N_I1', value=10.0, min=1.0, max=1000.0)
    # Step 2: I1 -> I2 (Medium: 1s - 1000s)
    params.add('k2_I1_I2', value=0.1, min=0.001, max=1.0)
    # Step 3: I2 -> U (Slow: > 1000s)
    params.add('k3_I2_U', value=0.0001, min=1e-7, max=0.001)

    # --- PATH 2 (Assume this is the CD-dominant slow path) ---
    # Step 1: N -> I3 (Fast: < 1s)
    params.add('k4_N_I3', value=5.0, min=1.0, max=1000.0)
    # Step 2: I3 -> I4 (Medium: 1s - 1000s)
    params.add('k5_I3_I4', value=0.05, min=0.001, max=1.0)
    # Step 3: I4 -> U (Very Slow: Hours)
    params.add('k6_I4_U', value=1e-5, min=1e-8, max=1e-4)

    # --- SPECTROSCOPIC FACTORS ---
    # We must constrain these so the intermediates don't just equal Native (invisible)

    # SF (Volts): N=High, U=Low
    # Anchor Start/End
    params.add('spec_sf_N', value=avg_sf_start, vary=False)  # Locked
    params.add('spec_sf_U', value=avg_sf_end, vary=False)  # Locked
    # Allow intermediates to float between N and U
    mid_sf = (avg_sf_start + avg_sf_end) / 2
    for s in ['I1', 'I2', 'I3', 'I4']:
        params.add(f'spec_sf_{s}', value=mid_sf, min=min(avg_sf_start, avg_sf_end), max=max(avg_sf_start, avg_sf_end))

    # CD (mdeg): N=High(0), U=Low(neg)
    # Anchor Start/End
    params.add('spec_cd_N', value=avg_cd_start, vary=False)  # Locked
    params.add('spec_cd_U', value=avg_cd_end, vary=False)  # Locked
    # Allow intermediates to float
    mid_cd = (avg_cd_start + avg_cd_end) / 2
    for s in ['I1', 'I2', 'I3', 'I4']:
        params.add(f'spec_cd_{s}', value=mid_cd, min=min(avg_cd_start, avg_cd_end), max=max(avg_cd_start, avg_cd_end))

    # --- FIT ---
    print(f"\nRunning PARALLEL Fit ({len(data_blocks)} sets)...")
    minner = Minimizer(objective, params, fcn_args=(data_blocks,))

    # Changed maxfev to max_nfev
    result = minner.minimize(method='leastsq', max_nfev=10000)

    # --- REPORT ---
    print("\n" + "=" * 50 + "\n RESULTS: PARALLEL MODEL\n" + "=" * 50)


    def print_k(name, p_key):
        k = result.params[p_key].value
        err = result.params[p_key].stderr or 0
        if k > 1e-12:
            th = np.log(2) / k
            th_err = (np.log(2) / k ** 2) * err
            unit = "s" if th <= 14400 else "h"
            val = th if th <= 14400 else th / 3600
            err_val = th_err if th <= 14400 else th_err / 3600
            print(f"{name}: {val:.2f} ± {err_val:.2f} {unit}")
        else:
            print(f"{name}: [Rate -> 0]")


    print("--- Path 1 (N -> I1 -> I2 -> U) ---")
    print_k("Step 1 (N->I1)", 'k1_N_I1')
    print_k("Step 2 (I1->I2)", 'k2_I1_I2')
    print_k("Step 3 (I2->U) ", 'k3_I2_U')

    print("\n--- Path 2 (N -> I3 -> I4 -> U) ---")
    print_k("Step 1 (N->I3)", 'k4_N_I3')
    print_k("Step 2 (I3->I4)", 'k5_I3_I4')
    print_k("Step 3 (I4->U) ", 'k6_I4_U')

    # --- PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    k_names = ['k1_N_I1', 'k2_I1_I2', 'k3_I2_U', 'k4_N_I3', 'k5_I3_I4', 'k6_I4_U']
    k_fin = {k: result.params[k].value for k in k_names}
    s_sf = [result.params[f'spec_sf_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'U']]
    s_cd = [result.params[f'spec_cd_{s}'].value for s in ['N', 'I1', 'I2', 'I3', 'I4', 'U']]

    # SF
    for b in data_blocks:
        if b['type'] == 'SF':
            off = result.params[b['offset_key']].value
            t_sm = np.geomspace(max(1e-4, min(b['t'])), max(b['t']), 1000)
            model = solve_mechanism(t_sm, k_fin, s_sf, off, 1.0)  # Scale fixed 1.0
            ax1.plot(b['t'], b['y'], 'o', color='lightgray', ms=2)
            ax1.plot(t_sm, model, 'r-', lw=1)
    ax1.set_xscale('linear');
    ax1.set_xlabel("Time (s)");
    ax1.set_title("Parallel Fit: SF")

    # CD
    ax_ins = ax2.inset_axes([0.45, 0.08, 0.50, 0.35])
    max_t = max([max(b['t']) for b in data_blocks if b['type'] == 'CD'])
    for b in data_blocks:
        if b['type'] == 'CD':
            off = result.params[b['offset_key']].value
            col = 'blue' if 'Spectra' in b['name'] else 'green'
            t_sm = np.linspace(min(b['t']), max(b['t']), 1000)
            model = solve_mechanism(t_sm, k_fin, s_cd, off, 1.0)
            ax2.plot(b['t'] / 3600, b['y'], 'o', color=col, ms=3, alpha=0.3)
            ax2.plot(t_sm / 3600, model, 'k--', lw=1.5)
            ax_ins.plot(b['t'] / 3600, b['y'], 'o', color=col, ms=4, alpha=0.4)
            ax_ins.plot(t_sm / 3600, model, 'k--', lw=1.5)

    ax2.set_xlim(0, max_t / 3600 * 1.05);
    ax2.set_xlabel("Time (h)")
    ax_ins.set_xlim(-0.01, 2000 / 3600);
    ax2.indicate_inset_zoom(ax_ins, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/ODE_fits/par_model/ODE_fit2.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()