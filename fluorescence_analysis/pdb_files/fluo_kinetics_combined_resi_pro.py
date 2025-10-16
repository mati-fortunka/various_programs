import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re

# ------------------- Core Models -------------------

def exponential(t, A, k, c):
    return A * np.exp(-k * t) + c

def single_exponential_with_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t

def double_exponential(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c

def sigmoid_after_lag(t, L, k, x0, lag):
    return np.where(t > lag, L / (1 + np.exp(-k * (t - x0))), 0)

def double_exp_plus_sigmoid(t, A1, k1, A2, k2, c, L, ksig, x0, lag):
    return double_exponential(t, A1, k1, A2, k2, c) + sigmoid_after_lag(t, L, ksig, x0, lag)

# ------------------- Fit Functions -------------------

def fit_exponential(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [A0, 0.01, C]
    try:
        popt, pcov = curve_fit(exponential, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
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
        return None, None

def fit_double_exponential(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [0.7 * A0, 0.01, 0.3 * A0, 0.001, C]
    try:
        popt, pcov = curve_fit(double_exponential, time, intensity, p0=initial_guess, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        return None, None

def fit_double_exp_plus_sigmoid(time, intensity, lag_start=1500):
    A0 = max(intensity)
    C = min(intensity)
    initial_guess = [0.6 * A0, 0.01, 0.3 * A0, 0.001, C, 0.1*A0, 0.001, lag_start+100, lag_start]
    try:
        popt, pcov = curve_fit(double_exp_plus_sigmoid, time, intensity,
                               p0=initial_guess, maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        return None, None

def fit_linear(time, intensity):
    coeffs = np.polyfit(time, intensity, 1)
    return coeffs

def read_data_file(filepath, property_name="CircularDichroism"):
    """
    Reads either a standard CSV or a ProDataCSV file.
    Returns: time (np.array), intensity (np.array)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect ProDataCSV by presence of metadata header
    if any("Available Properties" in line for line in lines):
        # --- Parse ProDataCSV format ---
        data_section_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith(property_name):
                data_section_start = i
                break

        if data_section_start is None:
            raise ValueError(f"Property '{property_name}' not found in {filepath}")

        # Find where data begins (skip the next 2 header lines)
        data_start = data_section_start + 3
        data = []
        for line in lines[data_start:]:
            if line.strip() == "" or re.match(r"^[A-Za-z]", line.strip()):
                break  # End of this block
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    y = float(parts[1])
                    data.append((t, y))
                except ValueError:
                    pass

        df = pd.DataFrame(data, columns=["Time", "Intensity"])
        return df["Time"].values, df["Intensity"].values

    else:
        # --- Standard CSV format (your current format) ---
        df = pd.read_csv(filepath, skiprows=2)
        df = df.dropna(how='all', axis=1)
        time = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        return time, intensity

# ------------------- Plotting -------------------

def plot_multiple_csvs_with_logging(folder_path,
                                    smooth_method='savitzky_golay',
                                    window_size=15,
                                    polyorder=3,
                                    dead_time=30,
                                    fit_type=None,
                                    fit_start=None,
                                    fit_end=None,
                                    output_plot="combined_plot.png",
                                    output_log="fitted_parameters.txt",
                                    lag_start=1000):

    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    color_cycle = plt.cm.hsv(np.linspace(0, 1, len(files)))

    fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios':[3,1]})

    param_log = []
    t_half_values = {"exp": [], "exp_drift": [], "k1": [], "k2": []}

    for i, filepath in enumerate(files):
        time, intensity = read_data_file(filepath, property_name="CircularDichroism")
        time = time + dead_time

        if smooth_method == 'moving_average':
            smoothed = np.convolve(intensity, np.ones(window_size)/window_size, mode='valid')
            time_plot = time[:len(smoothed)]
        elif smooth_method == 'savitzky_golay':
            smoothed = savgol_filter(intensity, window_size, polyorder)
            time_plot = time
        else:
            smoothed = intensity
            time_plot = time

        label_base = os.path.basename(filepath).replace(".csv", "")
        ax_main.plot(time_plot, smoothed, label=label_base, color=color_cycle[i])

        if fit_type:
            fit_time = time
            fit_intensity = intensity
            if fit_start is not None and fit_end is not None:
                mask = (time >= fit_start) & (time <= fit_end)
                fit_time = time[mask]
                fit_intensity = intensity[mask]

            line_log = f"{label_base} - "

            if fit_type == 'exponential':
                params, errors = fit_exponential(fit_time, fit_intensity)
                if params is not None:
                    A, k, c = params
                    eA, ek, ec = errors
                    fit_vals = exponential(fit_time, *params)
                    ax_main.plot(fit_time, fit_vals, linestyle='dotted', color=color_cycle[i])
                    ax_resid.plot(fit_time, fit_intensity - fit_vals, color=color_cycle[i])
                    t_half = np.log(2)/k if k > 0 else np.nan
                    t_half_values["exp"].append(t_half)
                    line_log += f"Exp Fit: A={A:.4f}±{eA:.4f}, k={k:.6f}±{ek:.6f}, c={c:.4f}±{ec:.4f}, t_half={t_half:.2f}"

            elif fit_type == 'exponential_with_drift':
                params, errors = fit_exponential_with_drift(fit_time, fit_intensity)
                if params is not None:
                    A, k, c, m = params
                    eA, ek, ec, em = errors
                    fit_vals = single_exponential_with_drift(fit_time, *params)
                    ax_main.plot(fit_time, fit_vals, linestyle='dotted', color=color_cycle[i])
                    ax_resid.plot(fit_time, fit_intensity - fit_vals, color=color_cycle[i])
                    t_half = np.log(2)/k if k > 0 else np.nan
                    t_half_values["exp_drift"].append(t_half)
                    line_log += f"Exp+Drift Fit: A={A:.4f}, k={k:.6f}, c={c:.4f}, m={m:.6f}, t_half={t_half:.2f}"

            elif fit_type == 'double_exponential':
                params, errors = fit_double_exponential(fit_time, fit_intensity)
                if params is not None:
                    A1, k1, A2, k2, c = params
                    eA1, ek1, eA2, ek2, ec = errors
                    fit_vals = double_exponential(fit_time, *params)
                    ax_main.plot(fit_time, fit_vals, linestyle='dotted', color=color_cycle[i])
                    ax_resid.plot(fit_time, fit_intensity - fit_vals, color=color_cycle[i])
                    t_half1 = np.log(2)/k1 if k1 > 0 else np.nan
                    t_half2 = np.log(2)/k2 if k2 > 0 else np.nan
                    t_half_values["k1"].append(t_half1)
                    t_half_values["k2"].append(t_half2)
                    frac = A1/(A1+A2) if (A1+A2)!=0 else np.nan
                    line_log += f"Double Exp Fit: A1={A1:.4f}±{eA1:.4f}, k1={k1:.6f}±{ek1:.6f}, A2={A2:.4f}±{eA2:.4f}, k2={k2:.6f}±{ek2:.6f}, c={c:.4f}±{ec:.4f}, t_half1={t_half1:.2f} s, t_half2={t_half2:.2f} s, A1/(A1+A2)={frac:.3f}"

            elif fit_type == 'double_exp_plus_sigmoid':
                params, errors = fit_double_exp_plus_sigmoid(fit_time, fit_intensity, lag_start=lag_start)
                if params is not None:
                    A1, k1, A2, k2, c, L, ksig, x0, lag = params
                    fit_vals = double_exp_plus_sigmoid(fit_time, *params)
                    ax_main.plot(fit_time, fit_vals, linestyle='dotted', color=color_cycle[i])
                    ax_resid.plot(fit_time, fit_intensity - fit_vals, color=color_cycle[i])
                    t_half1 = np.log(2)/k1 if k1 > 0 else np.nan
                    t_half2 = np.log(2)/k2 if k2 > 0 else np.nan
                    t_half_values["k1"].append(t_half1)
                    t_half_values["k2"].append(t_half2)
                    frac = A1/(A1+A2) if (A1+A2)!=0 else np.nan
                    line_log += f"Double Exp+Sigmoid Fit: A1={A1:.4f}, k1={k1:.6f}, A2={A2:.4f}, k2={k2:.6f}, c={c:.4f}, L={L:.4f}, ksig={ksig:.6f}, x0={x0:.2f}, lag={lag:.2f}, t_half1={t_half1:.2f}, t_half2={t_half2:.2f}, A1/(A1+A2)={frac:.3f}"

            elif fit_type == 'linear':
                slope, intercept = fit_linear(fit_time, fit_intensity)
                fit_vals = slope * fit_time + intercept
                ax_main.plot(fit_time, fit_vals, linestyle='dashed', color=color_cycle[i])
                ax_resid.plot(fit_time, fit_intensity - fit_vals, color=color_cycle[i])
                line_log += f"Linear Fit: slope={slope:.4f}, intercept={intercept:.4f}"

            param_log.append(line_log)

    with open(os.path.join(folder_path, output_log), "w") as f:
        f.write("\n".join(param_log))
        f.write("\n\nSummary Statistics:\n")
        for key, values in t_half_values.items():
            arr = np.array(values)
            if len(arr) > 0:
                avg, std = np.nanmean(arr), np.nanstd(arr)
                f.write(f"{key} t_half: mean={avg:.2f}, std={std:.2f}\n")

    ax_main.set_ylabel("Fluorescence Intensity (a.u.)")
    ax_main.set_title("Smoothed Fluorescence Curves with Fits")
    ax_main.legend(loc='best', fontsize='small')
    ax_main.grid(True)

    ax_resid.axhline(0, color='black', linewidth=0.8)
    ax_resid.set_xlabel("Time (s)")
    ax_resid.set_ylabel("Residuals")
    ax_resid.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, output_plot))
    plt.show()

    print(f"Results saved in {output_log}")
    print("\n".join(param_log))



if __name__ == "__main__":
    folder = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/kinetics/F8_september/2000s"

    plot_multiple_csvs_with_logging(
        folder_path=folder,
        smooth_method='savitzky_golay',
        window_size=15,
        polyorder=3,
        dead_time=30,
        fit_type='single_exponential',
        fit_start=0,
        fit_end=1500,
        output_plot="combined_kinetics_plot.png",
        output_log=f"{folder}/fitted_parameters.txt"
    )
