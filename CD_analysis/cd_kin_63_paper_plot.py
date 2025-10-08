import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob

wavelength_label = "Wavelength 218 nm"

def read_data(filename):
    global wavelength_label
    with open(filename, 'r') as file:
        lines = file.readlines()

    if lines[0].strip() == "ProDataCSV":
        print(f"🧠 ProDataCSV structure detected in '{filename}'")

        # Find where the CircularDichroism section starts
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("❌ Could not locate 'CircularDichroism' section in the file.")

        # Look for the data header ("Time", "Wavelength", etc.)
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip() == "":
                continue
            if "Time" in lines[j]:
                data_start_idx = j
                break
        else:
            raise ValueError("❌ Could not find data header after 'CircularDichroism'.")

        data_lines = lines[data_start_idx:]
        print(data_lines)
        data_str = ''.join(data_lines)
        from io import StringIO
        df_full = pd.read_csv(StringIO(data_str), sep=",")

        # Drop empty columns if any
        df_full = df_full.dropna(how='all', axis=1)

        # Handle time wrap (if time resets or decreases)
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

        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df.drop_duplicates(subset=df.columns[0], keep='first')
        return df

    else:
        print(f"📄 Standard CSV structure detected in '{filename}'")
        # Fallback to original logic
        first_line = lines[0].strip()
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
              dead_time=20, fit_type='exponential',
              fit_start=None, fit_end=None):
    """Process a single dataset: smooth + fit, but do NOT generate individual plots."""
    global wavelength_label

    time = pd.to_numeric(df.iloc[:, 0], errors="coerce") + file_dead_time
    cd_signal = df.iloc[:, 1]

    smoothed_time = None
    smoothed_signal = None

    if smooth_method == 'moving_average':
        smoothed_signal = moving_average(cd_signal, window_size)
        smoothed_time = time[:len(smoothed_signal)]
    elif smooth_method == 'savitzky_golay':
        smoothed_signal = savgol_filter(cd_signal, window_size, polyorder)
        smoothed_time = time

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
    fitted_values = None

    if fit_type == 'exponential':
        params, errors = fit_exponential(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1]
            fitted_values = exponential(fit_time, *params)
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
            fit_result = (
                f"Double exp fit: A1={params[0]:.7f}±{errors[0]:.7f}, "
                f"k1={params[1]:.7f}±{errors[1]:.7f}, A2={params[2]:.7f}±{errors[2]:.7f}, "
                f"k2={params[3]:.7f}±{errors[3]:.7f}, c={params[4]:.7f}±{errors[4]:.7f}, "
                f"t_half_k1={t_half_k1:.2f}s, t_half_k2={t_half_k2:.2f}s"
            )

    elif fit_type == 'linear':
        slope, intercept = fit_linear(fit_time, fit_cd)
        fitted_values = slope * fit_time + intercept
        fit_result = f"Linear fit: slope={slope:.7f}, intercept={intercept:.7f}"

    return fit_result, smoothed_time, smoothed_signal, fit_time, fitted_values


# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/CD/cd_kin_2000s_final/simple"
    smooth_method = 'savitzky_golay'
    window_size = 5
    polyorder = 3
    default_dead_time = 0
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)
    fit_type = "double_exponential"   # choose fit type if needed
    fit_start = 0
    fit_end = 2000

    results = []
    all_fit_params = []
    combined_curves = []
    fitted_curves = []
    raw_curves = []  # optional raw curves storage

    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)

            label = f"{filename_base}_{file_dead_time}"

            time = df.iloc[:, 0] + file_dead_time
            cd_signal = df.iloc[:, 1]

            raw_curves.append((time, cd_signal, str(label)))  # stored but not plotted

            fit_summary, smoothed_time, smoothed_signal, fit_time, fit_vals = plot_data(
                df,
                smooth_method=smooth_method,
                window_size=window_size,
                polyorder=polyorder,
                dead_time=file_dead_time,
                fit_type=fit_type,
                fit_start=fit_start,
                fit_end=fit_end
            )

            if smoothed_time is not None and smoothed_signal is not None:
                combined_curves.append((smoothed_time, smoothed_signal, str(label)))
            if fit_time is not None and fit_vals is not None:
                fitted_curves.append((fit_time, fit_vals, str(label)))

            results.append(f"{os.path.basename(filepath)}: {fit_summary}")

            param_matches = re.findall(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)±", fit_summary)
            if param_matches:
                all_fit_params.append([float(val) for val in param_matches])

        except Exception as e:
            error_msg = f"{os.path.basename(filepath)}: Failed to process - {str(e)}"
            print(error_msg)
            results.append(error_msg)

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
            for param_label, mean, std in zip(labels, means, stds):
                f.write(f"{param_label}: mean={mean:.7f}, std={std:.7f}\n")

        # --- Final combined plot of smoothed data (+ fits if requested) ---
        if combined_curves:
            plt.figure(figsize=(6, 5))
            color_map = plt.get_cmap('tab10')

            if fit_type:  # smoothed + fits
                for idx, ((stime, ssignal, label), (ftime, fsignal, _)) in enumerate(
                        zip(combined_curves, fitted_curves)):
                    color = "#3D48A4"
                    # color = color_map(idx % 10)
                    # Uncomment to show raw traces:
                    # rtime, rsignal, _ = raw_curves[idx]
                    # plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.3, color=color)

                    plt.plot(stime, ssignal, label=f"{label} (Smoothed)", color=color)
                    plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5, color=color)

            else:  # smoothed only
                for idx, (stime, ssignal, label) in enumerate(combined_curves):
                    color = "#3D48A4"
                    # color = color_map(idx % 10)
                    # Uncomment to show raw traces:
                    # rtime, rsignal, _ = raw_curves[idx]
                    # plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.3, color=color)
                    plt.plot(stime, ssignal, label=f"{label} (Smoothed)", color=color)

            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Ellipticity at 218 nm [mdeg]', fontsize=16)
            plt.xlim(0, 2000)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # plt.legend(fontsize=14, frameon=False)  # enable if needed
            plt.tight_layout()

            png_path = os.path.join(folder_path, "CD_SI_plot.png")
            svg_path = os.path.join(folder_path, "CD_SI_plot.svg")
            plt.savefig(png_path, dpi=600)
            plt.savefig(svg_path, dpi=600)

            plt.show()
            plt.close()

    # color_map = plt.get_cmap('tab10')


    """
    # Plot combined raw + fitted curves
    if raw_curves and fitted_curves:
        plt.figure(figsize=(6, 5))

        for idx, ((rtime, rsignal, label), (ftime, fsignal, _)) in enumerate(zip(raw_curves, fitted_curves)):
            color = color_map(idx % 10)  # Cycle through first 10 colors
            plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.5, color=color)
            plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5, color=color)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Ellipticity at 218 nm [mdeg]', fontsize=16)
        # plt.title(f'Smoothed CD Kinetics Curves ({wavelength_label})')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.legend(fontsize=14, frameon=False)
        plt.tight_layout()

        png_path = os.path.join(folder_path, "CD_main_plot.png")
        svg_path = os.path.join(folder_path, "CD_main_plot.svg")
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path, dpi=600)

        plt.show()
        plt.close()

"""