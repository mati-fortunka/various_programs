import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
from collections import defaultdict

def read_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()
        time_unit = "unknown"
        if "Time (min)" in first_line or "Time (min)" in second_line:
            print("⏱ Time units detected as minutes. Will convert to seconds.")
            time_unit = "min"
            skiprows = 1 if not re.match(r'^[\d\.\-]', first_line.split(',')[0]) else 0
        elif "Time (s)" in first_line or "Time (s)" in second_line:
            print("✅ Time units confirmed as seconds (s).")
            time_unit = "s"
            skiprows = 1 if not re.match(r'^[\d\.\-]', first_line.split(',')[0]) else 0
        else:
            print("⚠️ Time units not clearly specified. Assuming seconds.")
            skiprows = 1 if not re.match(r'^[\d\.\-]', first_line.split(',')[0]) else 0

    skiprows = 2
    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
    df_full = df_full.dropna(how='all', axis=1)

    time_col = df_full.iloc[:, 0]
    if time_unit == "min":
        time_col *= 60

    initial_time = time_col.iloc[0]
    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()
    if wrap_index > 1:
        print(f"Detected secondary block starting at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

    df.iloc[:, 0] = time_col
    return df

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def sigmoid_model(t, y0, a, k, t_half):
    """Sigmoidal growth/decay model."""
    return y0 + a / (1 + np.exp(-k * (t - t_half)))

def exponential(t, A, k, c):
    return A * np.exp(-k * t) + c

def single_exponential_with_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t

def double_exponential(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c

def estimate_initial_k(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity - half_max).argmin()
    t_half = time.iloc[half_max_index]
    return 1 / t_half if t_half > 0 else 1

def fit_sigmoid(time, intensity):
    """Fit data to sigmoid_model."""
    try:
        initial_guess = [min(intensity), max(intensity) - min(intensity), 1.0, (time.iloc[0] + time.iloc[-1]) / 2]
        popt, pcov = curve_fit(sigmoid_model, time, intensity, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Sigmoid fit failed.")
        return None, None

def fit_exponential(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [A0, 0.01, C]
    try:
        popt, pcov = curve_fit(exponential, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential fit failed.")
        return None, None

def fit_exponential_with_drift(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [A0, 0.01, C, 0.0]
    try:
        popt, pcov = curve_fit(single_exponential_with_drift, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential with drift fit failed.")
        return None, None

def fit_double_exponential(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [0.7 * A0, 0.01, 0.3 * A0, 0.001, C]
    try:
        popt, pcov = curve_fit(double_exponential, time, intensity, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Double exponential fit failed.")
        return None, None

def fit_linear(time, intensity):
    coeffs = np.polyfit(time, intensity, 1)
    return coeffs

def read_dead_times(folder_path, default_dead_time):
    dead_times = {}
    dead_file = os.path.join(folder_path, "dead_times.txt")
    if not os.path.exists(dead_file):
        print("⚠️ No 'dead_times.txt' found. Using default dead time.")
        return dead_times

    with open(dead_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0]
                match = re.match(r'(\d+(\.\d+)?)[sS]?', parts[1])
                if match:
                    dead_times[key] = float(match.group(1))
                else:
                    print(f"⚠️ Invalid dead time format for '{key}': '{parts[1]}'")
    return dead_times

def plot_multiple_csvs_with_logging(
    folder_path,
    smooth_method='savitzky_golay',
    window_size=25,
    polyorder=3,
    dead_time=30,
    fit_type=None,
    fit_start=None,
    fit_end=None,
    fit_ranges=None,              # dict: {"filename.csv": (start, end), ...}
    fit_files=None,               # optional list of filenames (for parallel lists mode)
    fit_starts=None,              # list of start times
    fit_ends=None,                # list of end times
    output_plot="combined_plot.png",
    output_log="fitted_parameters.txt",
    x_limits=None,
    y_limits=None
):
    plt.figure(figsize=(6, 5))
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    color_map = plt.get_cmap('tab10')

    param_log = []
    t_half_values = defaultdict(list)

    dead_times_dict = read_dead_times(folder_path, dead_time)

    for i, filepath in enumerate(files):
        df = read_data(filepath)
        label_base = os.path.basename(filepath).replace(".csv", "")
        custom_dead_time = dead_times_dict.get(label_base, dead_time)
        print(f"{label_base}: using dead time {custom_dead_time} s")

        time = df.iloc[:, 0] + custom_dead_time
        intensity = df.iloc[:, 1]

        # color = color_map(i % 10)
        color = "#3D48A4"

        # --- Plot smoothed data ---
        if smooth_method == 'moving_average':
            smoothed = moving_average(intensity, window_size)
            time_adjusted = time[:len(smoothed)]
            plt.plot(time_adjusted, smoothed, label=f"{label_base}", color=color)
        elif smooth_method == 'savitzky_golay':
            smoothed = savgol_filter(intensity, window_size, polyorder)
            plt.plot(time, smoothed, label=f"{label_base}", color=color)
        else:
            plt.plot(time, intensity, label=f"{label_base} (raw)", linestyle='--', alpha=0.5, color=color)

        # --- Determine custom fit range ---
        this_fit_start, this_fit_end = fit_start, fit_end  # default/global

        # Option 1: dictionary
        if fit_ranges and f"{label_base}.csv" in fit_ranges:
            this_fit_start, this_fit_end = fit_ranges[f"{label_base}.csv"]

        # Option 2: parallel lists
        elif fit_files and fit_starts and fit_ends:
            if label_base + ".csv" in fit_files:
                idx = fit_files.index(label_base + ".csv")
                this_fit_start, this_fit_end = fit_starts[idx], fit_ends[idx]

        # Apply mask if valid range is set
        fit_time, fit_intensity = time, intensity
        if this_fit_start is not None and this_fit_end is not None:
            fit_start_adj = this_fit_start + custom_dead_time
            fit_end_adj = this_fit_end + custom_dead_time
            mask = (time >= fit_start_adj) & (time <= fit_end_adj)
            fit_time = time[mask]
            fit_intensity = intensity[mask]

        # --- Fits ---
        if fit_type:
            line_log = f"{label_base} - "

            if fit_type == 'exponential':
                params, errors = fit_exponential(fit_time, fit_intensity)
                if params is not None:
                    A, k, c = params
                    eA, ek, ec = errors
                    plt.plot(fit_time, exponential(fit_time, *params),
                             label=f"{label_base} Exp Fit", linestyle='--', linewidth=1.5, color=color)
                    t_half = np.log(2)/k if k > 0 else np.nan
                    line_log += f"Exp Fit: A={A:.4f}±{eA:.4f}, k={k:.6f}±{ek:.6f}, c={c:.4f}±{ec:.4f}, t_half={t_half:.2f} s"
                    t_half_values['exp'].append(t_half)

            elif fit_type == 'sigmoid':
                params, errors = fit_sigmoid(fit_time, fit_intensity)
                if params is not None:
                    y0, a, k, t_half = params
                    ey0, ea, ek, eth = errors
                    plt.plot(fit_time, sigmoid_model(fit_time, *params),
                             label=f"{label_base} Sigmoid Fit", linestyle='--', linewidth=1.5, color=color)
                    line_log += (f"Sigmoid Fit: y0={y0:.4f}±{ey0:.4f}, a={a:.4f}±{ea:.4f}, "
                                 f"k={k:.6f}±{ek:.6f}, t_half={t_half:.2f}±{eth:.2f}")
                    t_half_values['sigmoid'].append(t_half)

            # (keep your other fit models here...)

            param_log.append(line_log)

    # --- Save log ---
    with open(os.path.join(folder_path, output_log), "w") as f:
        f.write("\n".join(param_log) + "\n\n")
        f.write("Summary Statistics:\n")
        for k, t_list in t_half_values.items():
            t_array = np.array(t_list)
            avg = np.nanmean(t_array)
            std = np.nanstd(t_array)
            f.write(f"{k} t_half: mean = {avg:.2f} s, std = {std:.2f} s\n")

    if x_limits:
        plt.xlim(*x_limits)
    if y_limits:
        plt.ylim(*y_limits)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Fluorescence Intensity (a.u.)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, output_plot), dpi=600)
    plt.show()
    plt.close()
    print(f"\nCombined plot saved as: {output_plot}")
    print(f"Fit results and t_half values saved to: {output_log}")



if __name__ == "__main__":
    folder = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/fluo/new fits/sigmoid"

    plot_multiple_csvs_with_logging(
        folder_path=folder,
        smooth_method='savitzky_golay',
        window_size=15,
        polyorder=3,
        dead_time=0,
        fit_type="sigmoid",
        # fit_ranges=None,
        fit_files=["63_2h_3.csv", "63_2h_6.csv", "63_4h_7.csv", "63_12h_1.csv"],
        fit_starts = [1400, 1780, 1400, 1850],
        fit_ends = [1900, 2400, 1900, 2600],
        output_plot="SI_fluo_plot4.svg",
        output_log=f"{folder}/fitted_parameters.txt",
        x_limits=(1300, 2600),
        y_limits=(5,15)
    )
