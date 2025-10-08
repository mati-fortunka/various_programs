import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob

wavelength_label = "Wavelength 222 nm"
labels = ["alpha", "gamma", "zeta"]
colors = ["#75053b", "#136308", "#0721a6"]
label_color_map = dict(zip(labels, colors))

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
            print("\u26a0\ufe0f No header detected or malformed header. Assuming data starts immediately.")
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
        print(f"\u26a0\ufe0f Dead time file '{file_path}' not found. Using default dead time.")
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
                    print(f"\u26a0\ufe0f Invalid dead time format for file '{name}': '{time_str}'")
    return dead_times

def exponential(t, A, k, c):
    return A * np.exp(-k * t) + c

def estimate_initial_k(time, intensity):
    A0 = max(intensity)
    C = min(intensity)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity - half_max).argmin()
    t_half = time.iloc[half_max_index]
    return 1 / t_half if t_half > 0 else 1

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

# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/CD_kinetics/new_zeta"
    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    default_dead_time = 30
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)
    fit_type = 'exponential'
    fit_start = 0
    fit_end = 2000
    protein = "zeta"
    x_lim = (-1,2000)
    results = []
    all_fit_params = []
    fitted_curves = []
    raw_curves = []
    t_halves = []

    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)

            time = df.iloc[:, 0] + file_dead_time
            cd_signal = df.iloc[:, 1]
            raw_curves.append((time, cd_signal, filename_base))

            fit_mask = (time >= time.iloc[0] + fit_start) & (time <= time.iloc[0] + fit_end)
            fit_time = time[fit_mask]
            fit_cd = cd_signal[fit_mask]

            params, errors = fit_exponential(fit_time, fit_cd)
            if params is not None:
                fit_vals = exponential(fit_time, *params)
                fitted_curves.append((fit_time, fit_vals, filename_base))
                t_half = np.log(2) / params[1]
                t_halves.append(t_half)
                fit_summary = (
                    f"Exponential fit: A={params[0]:.7f}±{errors[0]:.7f}, "
                    f"k={params[1]:.7f}±{errors[1]:.7f}, c={params[2]:.7f}±{errors[2]:.7f}, "
                    f"t_half={t_half:.2f}s"
                )
                results.append(f"{os.path.basename(filepath)}: {fit_summary}")

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
            labels = ['A', 'k', 'c']
            f.write("\n\nSummary of Fitted Parameters:\n")
            for label, mean, std in zip(labels, means, stds):
                f.write(f"{label}: mean={mean:.7f}, std={std:.7f}\n")
            f.write(f"t_half: mean={np.mean(t_halves):.7f}, std={np.std(t_halves):.7f}")

    # Plot combined raw + fitted curves
    if raw_curves and fitted_curves:
        plt.figure(figsize=(6, 5))
        for (rtime, rsignal, label), (ftime, fsignal, _) in zip(raw_curves, fitted_curves):
            color = label_color_map.get(protein, 'gray')
            plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.5, color=color)
            plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5, color=color)

        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Ellipticity at 222 nm [mdeg]', fontsize=16)
        if x_lim:
            plt.xlim(*x_lim)
        locs, labels = plt.xticks()
        plt.xticks(locs[1::2], labels[1::2], fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        png_path = os.path.join(folder_path, "combined_raw_fitted_plot.png")
        svg_path = os.path.join(folder_path, "combined_raw_fitted_plot.svg")
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path, dpi=600)
        plt.show()
        plt.close()
