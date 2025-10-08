#!/usr/bin/env python3
"""
combine_cd_runs.py

Read multiple CD CSV files from a folder (each file with CircularDichroism and HV sections),
extract:
 - Plot 2: CD at specific wavelength vs time (smoothed trace)
 - Plot 3: Integrated CD between a wavelength range vs time (smoothed + integration)

Combine results from all runs on two plots. Optionally fit (linear / single / double)
the time series for each run. Save PNG and SVG outputs.

Author: generated for user's request
"""

import os
import glob
from io import StringIO
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # identity iterator if tqdm not installed

# --------------------------
# Parameters (clean config)
# --------------------------
class Params:
    # Input directory containing CSV files
    input_dir: str = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/CD/spectra-kin"
    # Optional native spectrum (two-column text file, tab or whitespace separated)
    native_spectrum: Optional[str] = None  # e.g. "/path/to/native.txt"
    # Basic analysis options
    protein_label: str = "zeta"
    fit_model: Optional[str] = None  # "double" / "single" / "linear" / None
    hv_threshold: float = 900.0
    dead_time: float = 30.0  # seconds
    nm_per_sec: float = 0.4
    smoothing_window: int = 15  # will be forced odd
    smoothing_polyorder: int = 3
    baseline_correction: bool = False
    baseline_wavelength: float = 250.0
    target_wavelength: float = 218.0
    integration_range: Tuple[float, float] = (214.0, 250.0)
    integration_sign: str = "negative"  # "negative" or "positive"
    transpose_data: bool = True
    output_prefix: str = "Combined_CD_SK"
    show_legend: bool = False  # you said you don't need legend, default False
    save_svg: bool = True
    save_png: bool = True


# --------------------------
# Helpers
# --------------------------
def extract_section(lines: List[str], section_name: str) -> List[str]:
    """
    Find section header equals exact `section_name`, skip the next line (assumed header),
    and return the contiguous non-empty lines following as section content.
    """
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    else:
        raise ValueError(f"Section {section_name} not found in file")
    section_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        section_lines.append(line)
    return section_lines


def safe_make_odd(n: int) -> int:
    n = max(3, int(n))
    return n if (n % 2 == 1) else n + 1


def is_float_string(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def print_fit_params(popt: np.ndarray, pcov: Optional[np.ndarray], param_names: List[str]) -> None:
    print("  Fitted parameters:")
    for i, name in enumerate(param_names):
        error = np.sqrt(np.diag(pcov))[i] if (pcov is not None and pcov.shape[0] > i) else float('nan')
        print(f"    {name} = {popt[i]:.6g} ± {error:.6g}")


# Fit models
def linear(t, k, b):
    return k * t + b


def single_exp(t, a, k, c):
    return a * np.exp(-k * t) + c


def double_exp(t, a, k1, c, k2, e):
    return a * np.exp(-k1 * t) + c * np.exp(-k2 * t) + e


# --------------------------
# Core file processing
# --------------------------
def read_cd_hv_from_csv(filepath: str, params: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read CSV file and extract CircularDichroism and HV sections as DataFrames.
    Keeps original orientation until we optionally transpose later.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    cd_lines = extract_section(lines, "CircularDichroism")
    hv_lines = extract_section(lines, "HV")

    cd_df = pd.read_csv(StringIO(''.join(cd_lines)), skipinitialspace=True)
    hv_df = pd.read_csv(StringIO(''.join(hv_lines)), skipinitialspace=True)

    # rename first column to "Wavelength" for consistency
    cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
    hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)

    # drop empty columns
    cd_df.dropna(axis=1, how='all', inplace=True)
    hv_df.dropna(axis=1, how='all', inplace=True)

    return cd_df, hv_df


def transpose_and_align(cd_df: pd.DataFrame, hv_df: pd.DataFrame, params: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If params.transpose_data is True, convert data where original orientation was:
    first column = times, columns = wavelengths -> transform into rows = wavelengths, columns = times.
    Return cd_df, hv_df where first column is 'Wavelength' and subsequent columns are times (floats).
    """
    if not params.transpose_data:
        # Ensure first column is Wavelength and subsequent columns are numeric time columns
        # No transposition; ensure numeric column names where possible
        # Keep as-is
        return cd_df, hv_df

    # CD transpose
    cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce').values
    cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce').values
    cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
    cd_new = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
    cd_new.rename(columns={'index': 'Wavelength'}, inplace=True)
    cd_new = cd_new.dropna(subset=['Wavelength']).sort_values('Wavelength').reset_index(drop=True)

    # HV transpose
    hv_times = pd.to_numeric(hv_df.iloc[:, 0], errors='coerce').values
    hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors='coerce').values
    hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
    hv_new = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times).reset_index()
    hv_new.rename(columns={'index': 'Wavelength'}, inplace=True)
    hv_new = hv_new.dropna(subset=['Wavelength']).sort_values('Wavelength').reset_index(drop=True)

    # Align wavelengths (keep common intersection)
    common_wl = np.intersect1d(cd_new['Wavelength'].values, hv_new['Wavelength'].values)
    cd_new = cd_new[cd_new['Wavelength'].isin(common_wl)].reset_index(drop=True)
    hv_new = hv_new[hv_new['Wavelength'].isin(common_wl)].reset_index(drop=True)

    # Convert column names for times to floats (skip first Wavelength column)
    def col_map(df):
        return {float(col): col for col in df.columns if col != 'Wavelength'}

    # Return transformed
    return cd_new, hv_new


def build_col_map_from_df(df: pd.DataFrame) -> Dict[float, str]:
    """
    Map numeric time (float) -> original column name.
    Skip 'Wavelength' and any Unnamed columns.
    """
    col_map = {}
    for col in df.columns:
        if col == 'Wavelength' or str(col).startswith("Unnamed"):
            continue
        # if the column label is numeric or parseable to float, use it
        try:
            key = float(col)
            col_map[key] = col
        except Exception:
            # skip non-numeric column names
            continue
    return col_map


def process_file(filepath: str, params: Params) -> Dict:
    """
    Read file and compute:
      - times_hr_plot2, cd_values_at_target
      - times_hr_plot3, integrated_cd_values
    Return a dict with results and metadata.
    """
    fname = os.path.basename(filepath)
    print(f"\n🔁 Processing {fname}")

    cd_df, hv_df = read_cd_hv_from_csv(filepath, params)
    cd_df, hv_df = transpose_and_align(cd_df, hv_df, params)

    # Build column maps
    cd_col_map = build_col_map_from_df(cd_df)
    hv_col_map = build_col_map_from_df(hv_df)

    if not cd_col_map:
        raise ValueError(f"No usable time columns found in CD section of {fname}")
    if not hv_col_map:
        raise ValueError(f"No usable time columns found in HV section of {fname}")

    cd_times = sorted(cd_col_map.keys())
    hv_times = sorted(hv_col_map.keys())

    wavelengths = pd.to_numeric(cd_df['Wavelength']).values

    # smoothing window odd
    smooth_w = safe_make_odd(params.smoothing_window)

    # Find target wavelength index
    target_idx = np.argmin(np.abs(wavelengths - params.target_wavelength))
    actual_target_wl = wavelengths[target_idx]

    # effective dead time for plot2 (use actual_target_wl)
    extra_dead_time = (params.baseline_wavelength - actual_target_wl) / params.nm_per_sec
    effective_dead_time_plot2 = params.dead_time + extra_dead_time

    # Prepare outputs
    plot2_times_s = []
    plot2_values = []
    plot3_times_s = []
    plot3_values = []

    # Time = 0 native (optional) will be added outside in main
    # Iterate through CD columns (times)
    for cd_time in cd_times:
        cd_col = cd_col_map.get(cd_time)
        # find closest hv time and corresponding col
        hv_time = min(hv_times, key=lambda t: abs(t - cd_time))
        hv_col = hv_col_map.get(hv_time)

        cd_curve = pd.to_numeric(cd_df[cd_col], errors='coerce').values
        hv_curve = pd.to_numeric(hv_df[hv_col], errors='coerce').values

        # guard: ensure same length
        if len(hv_curve) != len(wavelengths):
            # try to truncate/pad hv
            hv_curve = hv_curve[:len(wavelengths)]

        # HV mask
        hv_mask = hv_curve <= params.hv_threshold
        x_wl_masked = wavelengths[hv_mask]
        # ---- Plot 2 value at target wavelength (within hv mask) ----
        if target_idx < len(wavelengths):
            # find index in masked x closest to target
            # If target not present in masked x (because hv masked it out), skip
            if np.any(hv_mask):
                # get cd value aligned to wavelengths
                cd_masked = cd_curve[hv_mask]
                # smoothing
                if len(cd_masked) >= smooth_w:
                    cd_masked_sm = savgol_filter(cd_masked, window_length=smooth_w, polyorder=params.smoothing_polyorder)
                else:
                    cd_masked_sm = cd_masked
                # baseline correction if requested (index in masked vector)
                if params.baseline_correction and len(x_wl_masked) > 0:
                    baseline_idx_local = np.argmin(np.abs(x_wl_masked - params.baseline_wavelength))
                    baseline_val = cd_masked_sm[baseline_idx_local]
                else:
                    baseline_val = 0.0
                # find closest index in x_wl_masked to actual target wl
                if len(x_wl_masked) > 0 and (np.min(np.abs(x_wl_masked - actual_target_wl)) < 1e6):
                    idx_local = np.argmin(np.abs(x_wl_masked - actual_target_wl))
                    cd_val = cd_masked_sm[idx_local] - baseline_val
                    if not np.isnan(cd_val):
                        adj_time = cd_time + effective_dead_time_plot2
                        plot2_times_s.append(adj_time)
                        plot2_values.append(cd_val)
        # ---- Plot 3 integration ----
        # local x = x_wl_masked (wavelengths allowed by HV)
        # restrict to integration_range
        lambda_min, lambda_max = params.integration_range
        in_range_mask_local = (x_wl_masked >= lambda_min) & (x_wl_masked <= lambda_max)
        x_local = x_wl_masked[in_range_mask_local]
        if x_local.size > 1:
            y_local = cd_curve[hv_mask][in_range_mask_local]
            # smoothing
            if len(y_local) >= smooth_w:
                y_local_sm = savgol_filter(y_local, window_length=smooth_w, polyorder=params.smoothing_polyorder)
            else:
                y_local_sm = y_local
            # baseline correction
            if params.baseline_correction:
                base_idx = np.argmin(np.abs(x_local - params.baseline_wavelength))
                baseline_val = y_local_sm[base_idx]
                y_local_sm = y_local_sm - baseline_val
            # keep only sign-specified parts
            sign = params.integration_sign.lower()
            if sign == "positive":
                y_sign = np.where(y_local_sm > 0, y_local_sm, 0.0)
            else:
                # negative: keep negative values (so area will be negative or zero)
                y_sign = np.where(y_local_sm < 0, y_local_sm, 0.0)
            # integrate
            area = trapezoid(y_sign, x_local)
            time_s = cd_time + (params.dead_time + (params.baseline_wavelength - 0.5 * (lambda_min + lambda_max)) / params.nm_per_sec)
            plot3_times_s.append(time_s)
            plot3_values.append(area)

    # sort by time
    if plot2_times_s:
        order = np.argsort(plot2_times_s)
        plot2_times_s = list(np.array(plot2_times_s)[order])
        plot2_values = list(np.array(plot2_values)[order])
    if plot3_times_s:
        order3 = np.argsort(plot3_times_s)
        plot3_times_s = list(np.array(plot3_times_s)[order3])
        plot3_values = list(np.array(plot3_values)[order3])

    return {
        "filename": fname,
        "target_wavelength_actual": float(actual_target_wl),
        "plot2_times_s": np.array(plot2_times_s),
        "plot2_values": np.array(plot2_values),
        "plot3_times_s": np.array(plot3_times_s),
        "plot3_values": np.array(plot3_values),
    }


# --------------------------
# Plotting / main
# --------------------------
def fit_and_plot(ax, times_s, values, times_hr, color, label, params: Params):
    """
    Plot smoothed data as a curve (times_hr vs values) and optionally fit and plot fit.
    times_s: times in seconds (used by fit), times_hr is hours (for plotting x axis)
    """
    # plot smoothed line
    ax.plot(times_hr, values, color=color, linewidth=1.5)

    # optional fit
    if params.fit_model is None:
        return None

    t_fit = np.array(times_s)
    y_fit = np.array(values)
    if len(t_fit) < 4:
        print(f"   ⚠️ Not enough points to fit for {label}: {len(t_fit)} points")
        return None

    try:
        if params.fit_model == "linear":
            popt, pcov = curve_fit(linear, t_fit, y_fit)
            y_model = linear(t_fit, *popt)
            param_names = ['k', 'b']
        elif params.fit_model == "single":
            popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=(y_fit[0], 0.001, y_fit[-1]), maxfev=5000)
            y_model = single_exp(t_fit, *popt)
            param_names = ['a', 'k', 'c']
            try:
                t_half = log(2) / popt[1]
                print(f"   Half-life (s) = {t_half:.2f}")
            except Exception:
                pass
        elif params.fit_model == "double":
            p0 = (y_fit[0], 0.001, y_fit[0] / 2 if len(y_fit) > 1 else 0.0, 0.0001, y_fit[-1])
            popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, maxfev=8000)
            y_model = double_exp(t_fit, *popt)
            param_names = ['a', 'k1', 'c', 'k2', 'e']
            try:
                t_half1 = log(2) / popt[1]
                t_half2 = log(2) / popt[3]
                print(f"   Half-lives (s): {t_half1:.2f}, {t_half2:.2f}")
            except Exception:
                pass
        else:
            print(f"   ⚠️ Unknown fit_model: {params.fit_model}")
            return None

        # Sort for plotting smooth model curve
        idx = np.argsort(t_fit)
        t_sorted = t_fit[idx]
        y_model_sorted = y_model[idx]
        # convert seconds to hours for plotting
        t_sorted_hr = t_sorted / 3600.0
        ax.plot(t_sorted_hr, y_model_sorted, linestyle='--', color=color, linewidth=1.2)
        print_fit_params(popt, pcov, param_names)
        return popt
    except Exception as e:
        print(f"   ❌ Fit failed for {label}: {repr(e)}")
        return None


def main(params: Params):
    # prepare smoothing parameter
    params.smoothing_window = safe_make_odd(params.smoothing_window)

    # discover csv files
    pattern = os.path.join(params.input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {params.input_dir} (pattern: {pattern})")
    print(f"Found {len(files)} CSV files in {params.input_dir}")

    # optional native spectrum
    native_wl = native_cd = None
    if params.native_spectrum and os.path.exists(params.native_spectrum):
        try:
            nat = pd.read_csv(params.native_spectrum, sep=None, engine='python', header=None)
            native_wl = nat.iloc[:, 0].astype(float).values
            native_cd = nat.iloc[:, 1].astype(float).values
            print(f"Loaded native spectrum from {params.native_spectrum}")
        except Exception as e:
            print(f"Failed to load native spectrum: {e}")

    # process all files
    results = []
    for fpath in tqdm(files, desc="Files"):
        try:
            res = process_file(fpath, params)
            results.append(res)
        except Exception as e:
            print(f"Failed to process {os.path.basename(fpath)}: {e}")

    # plotting palette
    cmap = cm.get_cmap('tab10')
    # colors = [cmap(i % 10) for i in range(len(results))]
    color = "#3D48A4"

    # --- Combined Plot 2: CD at target wavelength vs time ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.set_xlabel("Time [h]", fontsize=15)
    ax2.set_ylabel(f"Ellipticity at {int(params.target_wavelength)} nm [mdeg]", fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    # ax2.set_title(f"Combined CD at {int(params.target_wavelength)} nm vs Time (hv ≤ {params.hv_threshold})")

    for i, r in enumerate(results):
        times_hr = r['plot2_times_s'] / 3600.0
        values = r['plot2_values']
        if len(times_hr) == 0:
            continue
        # color = colors[i]

        # plot smoothed raw (as line)
        ax2.plot(times_hr, values, color=color, linewidth=1.5)
        # plot fits if requested
        if params.fit_model is not None:
            print(f"\nFitting Plot2 for {r['filename']}")
            fit_and_plot(ax2, r['plot2_times_s'], r['plot2_values'], times_hr, color, r['filename'], params)

    # native at time 0 if available
    if native_wl is not None and native_cd is not None:
        # compute native value at target wl (optionally smoothed)
        native_y = native_cd.copy()
        if len(native_y) >= params.smoothing_window:
            native_y = savgol_filter(native_y, window_length=params.smoothing_window, polyorder=params.smoothing_polyorder)
        native_idx = np.argmin(np.abs(native_wl - params.target_wavelength))
        native_val = native_y[native_idx]
        ax2.scatter([0.0], [native_val], color='k', marker='x', s=50)

    if params.show_legend:
        ax2.legend([r['filename'] for r in results], fontsize=8)
    ax2.grid(False)
    plt.tight_layout()
    outbase2 = os.path.join(params.input_dir, f"{params.output_prefix}_at_{int(params.target_wavelength)}_nm_vs_time")
    if params.save_png:
        plt.savefig(outbase2 + ".png", dpi=300)
        print(f"Saved: {outbase2}.png")
    if params.save_svg:
        plt.savefig(outbase2 + ".svg")
        print(f"Saved: {outbase2}.svg")
    plt.show()

    # --- Combined Plot 3: Integrated CD vs time ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.set_xlabel("Time [h]", fontsize=15)
    ax3.set_ylabel("Integrated CD [mdeg·nm]", fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)
    lam_min, lam_max = params.integration_range
    # ax3.set_title(f"Combined Integrated CD ({params.integration_sign}) {lam_min}-{lam_max} nm vs Time")
    color = "#3D48A4"

    for i, r in enumerate(results):
        times_hr = r['plot3_times_s'] / 3600.0
        values = r['plot3_values']
        if len(times_hr) == 0:
            continue
        # color = colors[i]

        ax3.plot(times_hr, values, color=color, linewidth=1.5)
        if params.fit_model is not None:
            print(f"\nFitting Plot3 for {r['filename']}")
            fit_and_plot(ax3, r['plot3_times_s'], r['plot3_values'], times_hr, color, r['filename'], params)

    # native integrated at time 0
    if native_wl is not None and native_cd is not None:
        mask_nat = (native_wl >= lam_min) & (native_wl <= lam_max)
        if np.sum(mask_nat) > 1:
            x_nat = native_wl[mask_nat]
            y_nat = native_cd[mask_nat]
            if len(y_nat) >= params.smoothing_window:
                y_nat = savgol_filter(y_nat, window_length=params.smoothing_window, polyorder=params.smoothing_polyorder)
            if params.integration_sign.lower() == "positive":
                y_nat_sign = np.where(y_nat > 0, y_nat, 0.0)
            else:
                y_nat_sign = np.where(y_nat < 0, y_nat, 0.0)
            area_nat = trapezoid(y_nat_sign, x_nat)
            ax3.scatter([0.0], [area_nat], color='k', marker='x', s=50)

    if params.show_legend:
        ax3.legend([r['filename'] for r in results], fontsize=8)
    ax3.grid(False)
    plt.tight_layout()
    outbase3 = os.path.join(params.input_dir, f"{params.output_prefix}_integrated_{params.integration_sign}_{int(lam_min)}_{int(lam_max)}_vs_time")
    if params.save_png:
        plt.savefig(outbase3 + ".png", dpi=300)
        print(f"Saved: {outbase3}.png")
    if params.save_svg:
        plt.savefig(outbase3 + ".svg")
        print(f"Saved: {outbase3}.svg")
    plt.show()

    print("\n✅ Done.")


if __name__ == "__main__":
    # Edit parameters here before running
    p = Params()
    # Example: p.input_dir = "/path/to/your/csv_folder"
    # p.native_spectrum = "/path/to/native.txt"
    # p.fit_model = None  # if you want only smoothed curves without fits
    main(p)
