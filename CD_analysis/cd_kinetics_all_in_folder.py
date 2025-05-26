import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors

wavelength_label = "Wavelength 222 nm"

def read_data(filename):
    global wavelength_label
    with open(filename, 'r') as file:
        print(f"Processing {filename}...")
        first_line = file.readline().strip()
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            print(f"Header detected: '{first_line}'")
            match = re.search(r'(\d+(?:\.\d+)?)', first_line)
            if match:
                wavelength_label = f"Wavelength: {match.group(1)} nm"
            else:
                wavelength_label = first_line
            skiprows = 1
        else:
            print("⚠️ No header detected or malformed header. Assuming data starts immediately.")
            skiprows = 0

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
    df_full = df_full.dropna(how='all', axis=1)

    time_col = df_full.iloc[:, 0].values

    wrap_index = None
    for i in range(1, len(time_col)):
        if time_col[i] <= time_col[i - 1]:
            wrap_index = i
            break

    if wrap_index is not None:
        print(f"Detected time wrap at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

    df = df.drop_duplicates(subset=df.columns[0], keep='first')

    return df

def read_dead_times(file_path):
    dead_times = {}
    if not os.path.exists(file_path):
        print(f"⚠️ Dead time file '{file_path}' not found. Using default dead time.")
        return dead_times

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, time_str = parts
                try:
                    time_value = int(time_str.rstrip('s'))
                    dead_times[name] = time_value
                except ValueError:
                    print(f"⚠️ Invalid dead time format for file '{name}': '{time_str}'")
    return dead_times

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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
    k_init = 1 / t_half if t_half > 0 else 1
    return k_init

def fit_exponential(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
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
    k0 = estimate_initial_k(time, intensity)
    m0 = 0.0
    initial_guess = [A0, 0.01, C, m0]
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

def plot_data(df, smooth_method=None, window_size=5, polyorder=2,
              output_plot="out.png", dead_time=20,
              fit_type='exponential', fit_start=None, fit_end=None):
    global wavelength_label

    time = df.iloc[:, 0] + dead_time
    cd_signal = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(time, cd_signal, label='Raw CD Data', color='gray', alpha=0.6)

    smoothed_time = None
    smoothed_signal = None

    if smooth_method == 'moving_average':
        smoothed_signal = moving_average(cd_signal, window_size)
        smoothed_time = time[:len(smoothed_signal)]
        plt.plot(smoothed_time, smoothed_signal, label=f'Moving Avg (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed_signal = savgol_filter(cd_signal, window_size, polyorder)
        smoothed_time = time
        plt.plot(smoothed_time, smoothed_signal, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    # Define fit range relative to dead time
    if fit_start is None:
        fit_start_abs = time.iloc[0]
    else:
        fit_start_abs = time.iloc[0] + fit_start

    if fit_end is None:
        fit_end_abs = time.iloc[-1]
    else:
        fit_end_abs = time.iloc[0] + fit_end

    mask = (time >= fit_start_abs) & (time <= fit_end_abs)
    fit_time = time[mask]
    fit_cd = cd_signal[mask]

    fit_result = ""
    fitted_values = None  # NEW

    if fit_type == 'exponential':
        params, errors = fit_exponential(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1]
            fitted_values = exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exponential Fit', color='green')
            fit_result = (
                f"Exponential fit: A={params[0]:.7f}±{errors[0]:.7f}, "
                f"k={params[1]:.7f}±{errors[1]:.7f}, c={params[2]:.7f}±{errors[2]:.7f}, "
                f"t_half={t_half:.2f}s"
            )

    elif fit_type == 'exponential_drift':
        params, errors = fit_exponential_with_drift(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1]
            fitted_values = single_exponential_with_drift(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exp + Drift Fit', color='orange')
            fit_result = (
                f"Exp+Drift fit: A={params[0]:.7f}±{errors[0]:.7f}, "
                f"k={params[1]:.7f}±{errors[1]:.7f}, c={params[2]:.7f}±{errors[2]:.7f}, "
                f"m={params[3]:.7f}±{errors[3]:.7f}, t_half={t_half:.2f}s"
            )

    elif fit_type == 'double_exponential':
        params, errors = fit_double_exponential(fit_time, fit_cd)
        if params is not None:
            t_half_k1 = np.log(2) / params[1]
            t_half_k2 = np.log(2) / params[3]
            fitted_values = double_exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Double Exp Fit', color='brown')
            fit_result = (
                f"Double exp fit: A1={params[0]:.7f}±{errors[0]:.7f}, "
                f"k1={params[1]:.7f}±{errors[1]:.7f}, A2={params[2]:.7f}±{errors[2]:.7f}, "
                f"k2={params[3]:.7f}±{errors[3]:.7f}, c={params[4]:.7f}±{errors[4]:.7f}, "
                f"t_half_k1={t_half_k1:.2f}s, t_half_k2={t_half_k2:.2f}s"
            )

    elif fit_type == 'linear':
        slope, intercept = fit_linear(fit_time, fit_cd)
        fitted_values = slope * fit_time + intercept
        plt.plot(fit_time, fitted_values, label='Linear Fit', color='purple')
        fit_result = f"Linear fit: slope={slope:.7f}, intercept={intercept:.7f}"

    plt.xlabel('Time (s)')
    plt.ylabel('Ellipticity (mdeg)')
    plt.title(f'CD Kinetics Over Time\n({wavelength_label})')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.close()

    return fit_result, smoothed_time, smoothed_signal, fit_time, fitted_values

# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/kinetics/CD/2000s"
    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    default_dead_time = 30
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)
    fit_type = 'double_exponential'
    fit_start = 0
    fit_end = 2000

    results = []
    all_fit_params = []
    combined_curves = []
    fitted_curves = []
    raw_curves = []  # To store raw (time, signal, label)

    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)

            time = df.iloc[:, 0] + file_dead_time
            cd_signal = df.iloc[:, 1]
            raw_curves.append((time, cd_signal, filename_base))

            out_path = filepath[:-4] + "_fit.png"
            fit_summary, smoothed_time, smoothed_signal, fit_time, fit_vals = plot_data(
                df,
                smooth_method=smooth_method,
                window_size=window_size,
                polyorder=polyorder,
                output_plot=out_path,
                dead_time=file_dead_time,
                fit_type=fit_type,
                fit_start=fit_start,
                fit_end=fit_end
            )

            if smoothed_time is not None and smoothed_signal is not None:
                combined_curves.append((smoothed_time, smoothed_signal, filename_base))
            if fit_time is not None and fit_vals is not None:
                fitted_curves.append((fit_time, fit_vals, filename_base))

            results.append(f"{os.path.basename(filepath)}: {fit_summary}")

            if smoothed_time is not None and smoothed_signal is not None:
                combined_curves.append((smoothed_time, smoothed_signal, filename_base))

            param_matches = re.findall(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)±", fit_summary)
            if param_matches:
                all_fit_params.append([float(val) for val in param_matches])

        except Exception as e:
            results.append(f"{os.path.basename(filepath)}: Failed to process - {str(e)}")

    # Save fitting results
    with open(os.path.join(folder_path, "fitting_results.txt"), "w") as f:
        f.write("\n".join(results))

        if all_fit_params:
            param_array = np.array(all_fit_params)
            means = np.mean(param_array, axis=0)
            stds = np.std(param_array, axis=0)

            if fit_type == 'exponential':
                labels = ['A', 'k', 'c']
            elif fit_type == 'exponential_drift':
                labels = ['A', 'k', 'c', 'm']
            elif fit_type == 'double_exponential':
                labels = ['A1', 'k1', 'A2', 'k2', 'c']
            elif fit_type == 'linear':
                labels = ['slope', 'intercept']
            else:
                labels = [f"param{i + 1}" for i in range(param_array.shape[1])]

            f.write("\n\nSummary of Fitted Parameters:\n")
            for label, mean, std in zip(labels, means, stds):
                f.write(f"{label}: mean={mean:.7f}, std={std:.7f}\n")

    color_map = plt.get_cmap('tab10')
    # Plot combined raw + fitted curves
    if raw_curves and fitted_curves:
        plt.figure(figsize=(10, 6))
        for idx, ((rtime, rsignal, label), (ftime, fsignal, _)) in enumerate(zip(raw_curves, fitted_curves)):
            color = color_map(idx % 10)  # Cycle through first 10 colors
            plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.5, color=color)
            plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Ellipticity (mdeg)")
        plt.title(f"Combined Raw and Fitted CD Kinetics Curves\n({wavelength_label})")
        plt.legend(fontsize='small', loc='best')
        plt.grid(True)
        plt.tight_layout()
        combined_raw_fit_path = os.path.join(folder_path, "combined_raw_fitted_plot.png")
        plt.savefig(combined_raw_fit_path)
        plt.show()
        plt.close()

    """
    # Plot combined smoothed + fitted curves
    if combined_curves and fitted_curves:
        plt.figure(figsize=(10, 6))
        for (stime, ssignal, label), (ftime, fsignal, _) in zip(combined_curves, fitted_curves):
            plt.plot(stime, ssignal, label=f"{label} (Smoothed)")
            plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Ellipticity (mdeg)")
        plt.title(f"Combined Smoothed and Fitted CD Kinetics Curves\n({wavelength_label})")
        #plt.legend(fontsize='small', loc='best')
        plt.grid(True)
        plt.tight_layout()
        combined_fit_plot_path = os.path.join(folder_path, "combined_smoothed_fitted_plot.png")
        plt.savefig(combined_fit_plot_path)
        plt.show()
        plt.close()
        """