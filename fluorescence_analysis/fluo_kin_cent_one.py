import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
from collections import defaultdict


def read_data(filename):
    # --- 1. PRE-CHECK: Detect file format ---
    with open(filename, 'r') as f:
        header_lines = [f.readline() for _ in range(15)]

    is_new_format = any(line.startswith("Labels,") for line in header_lines)

    if is_new_format:
        print(f"🆕 Detected new format: {os.path.basename(filename)}")
        skip_rows = 0
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if re.match(r'^[\d]', line):
                    skip_rows = i
                    break
        try:
            df = pd.read_csv(filename, skiprows=skip_rows, header=None)
            df = df.dropna(how='all', axis=1)
            df = df.iloc[:, :2]
            df.iloc[:, 0] = df.iloc[:, 0] / 1e9
            print("✅ Converted time from nanoseconds to seconds.")
            return df
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return pd.DataFrame()
    else:
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            second_line = file.readline().strip()
            time_unit = "unknown"
            if "Time (min)" in first_line or "Time (min)" in second_line:
                print("⏱ Time units detected as minutes. Will convert to seconds.")
                time_unit = "min"
            elif "Time (s)" in first_line or "Time (s)" in second_line:
                print("✅ Time units confirmed as seconds (s).")
                time_unit = "s"
            else:
                print("⚠️ Time units not clearly specified. Assuming seconds.")

        skiprows = 2
        df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
        df_full = df_full.dropna(how='all', axis=1)

        time_col = df_full.iloc[:, 0]
        if time_unit == "min":
            time_col *= 60

        initial_time = time_col.iloc[0]
        wrap_index_series = time_col[1:].sub(initial_time).abs().lt(1e-6)
        if not wrap_index_series.empty and wrap_index_series.any():
            wrap_index = wrap_index_series.idxmax()
            if wrap_index > 1:
                print(f"Detected secondary block starting at row {wrap_index}. Truncating data.")
                df = df_full.iloc[:wrap_index]
            else:
                df = df_full
        else:
            df = df_full

        df.iloc[:, 0] = time_col
        return df


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def exponential(t, A, k, c):
    return A * np.exp(-k * t) + c


def single_exponential_with_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t


def double_exponential(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


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
        return dead_times

    with open(dead_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0]
                match = re.match(r'(\d+(\.\d+)?)[sS]?', parts[1])
                if match:
                    dead_times[key] = float(match.group(1))
    return dead_times


def plot_single_csv_with_fitting(file_path,
                                 smooth_method='savitzky_golay',
                                 window_size=25,
                                 polyorder=3,
                                 dead_time=30,
                                 fit_type=None,
                                 fit_start=None,
                                 fit_end=None):
    # Deriving directory and filenames dynamically based on the input file path
    folder_path = os.path.dirname(file_path)
    label_base = os.path.basename(file_path).replace(".csv", "")
    output_plot = os.path.join(folder_path, f"{label_base}_plot.png")
    output_log = os.path.join(folder_path, f"{label_base}_fit.txt")

    plt.figure(figsize=(10, 6))

    # Read dead time dictionary if it exists in the folder
    dead_times_dict = read_dead_times(folder_path, dead_time)
    custom_dead_time = dead_times_dict.get(label_base, dead_time)
    print(f"📊 Processing: {label_base} (Using dead time: {custom_dead_time} s)")

    df = read_data(file_path)
    if df.empty:
        print("❌ Could not read data file.")
        return

    time = df.iloc[:, 0] + custom_dead_time
    intensity = df.iloc[:, 1]

    # Plot original data or smoothed curves
    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f"{label_base} (MA Smoothed)", color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f"{label_base} (SG Smoothed)", color='blue')
    else:
        plt.plot(time, intensity, label=f"{label_base} (Raw)", linestyle='--', alpha=0.5, color='blue')

    line_log = f"{label_base} - "

    # Perform calculations and fitting if requested
    if fit_type:
        fit_time = time
        fit_intensity = intensity
        if fit_start is not None and fit_end is not None:
            fit_start_adj = fit_start + custom_dead_time
            fit_end_adj = fit_end + custom_dead_time
            mask = (time >= fit_start_adj) & (time <= fit_end_adj)
            fit_time = time[mask]
            fit_intensity = intensity[mask]

        if fit_type == 'exponential':
            params, errors = fit_exponential(fit_time, fit_intensity)
            if params is not None:
                A, k, c = params
                eA, ek, ec = errors
                plt.plot(fit_time, exponential(fit_time, *params), label="Exp Fit", linestyle='dotted', color='red',
                         linewidth=2)
                t_half = np.log(2) / k if k > 0 else np.nan
                line_log += f"Exp Fit: A={A:.4f}±{eA:.4f}, k={k:.6f}±{ek:.6f}, c={c:.4f}±{ec:.4f}, t_half={t_half:.2f} s"

        elif fit_type == 'exponential_with_drift':
            params, errors = fit_exponential_with_drift(fit_time, fit_intensity)
            if params is not None:
                A, k, c, m = params
                eA, ek, ec, em = errors
                plt.plot(fit_time, single_exponential_with_drift(fit_time, *params), label="Exp+Drift Fit",
                         linestyle='dotted', color='red', linewidth=2)
                t_half = np.log(2) / k if k > 0 else np.nan
                line_log += f"Exp+Drift Fit: A={A:.4f}±{eA:.4f}, k={k:.6f}±{ek:.6f}, c={c:.4f}±{ec:.4f}, m={m:.6f}±{em:.6f}, t_half={t_half:.2f} s"

        elif fit_type == 'double_exponential':
            params, errors = fit_double_exponential(fit_time, fit_intensity)
            if params is not None:
                A1, k1, A2, k2, c = params
                eA1, ek1, eA2, ek2, ec = errors
                plt.plot(fit_time, double_exponential(fit_time, *params), label="Double Exp Fit", linestyle='dotted',
                         color='red', linewidth=2)
                t_half1 = np.log(2) / k1 if k1 > 0 else np.nan
                t_half2 = np.log(2) / k2 if k2 > 0 else np.nan
                line_log += (f"Double Exp Fit: A1={A1:.4f}±{eA1:.4f}, k1={k1:.6f}±{ek1:.6f}, "
                             f"A2={A2:.4f}±{eA2:.4f}, k2={k2:.6f}±{ek2:.6f}, c={c:.4f}±{ec:.4f}, "
                             f"t_half1={t_half1:.2f} s, t_half2={t_half2:.2f} s")

        elif fit_type == 'linear':
            slope, intercept = fit_linear(fit_time, fit_intensity)
            plt.plot(fit_time, slope * fit_time + intercept, label="Linear Fit", linestyle='dashed', color='red',
                     linewidth=2)
            line_log += f"Linear Fit: slope={slope:.4f}, intercept={intercept:.4f}"

    # Save log parameters for the single file
    with open(output_log, "w") as f:
        f.write(line_log + "\n")

    # Plot labels and formatting
    plt.xlabel("Time (s)")
    plt.ylabel("Fluorescence Intensity (a.u.)")
    plt.title(f"Fluorescence Curve and Fit for {label_base}")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

    print(f"💾 Plot saved as: {output_plot}")
    print(f"📝 Fit results saved to: {output_log}")


if __name__ == "__main__":
    # Specify the direct path to your single CSV file here
    target_file = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/fuzja/best_all/Kinetic fuzja Scan1.csv"

    plot_single_csv_with_fitting(
        file_path=target_file,
        smooth_method='savitzky_golay',
        window_size=15,
        polyorder=3,
        dead_time=20,
        fit_type="exponential_with_drift",
        fit_start=0,
        fit_end=10000
    )