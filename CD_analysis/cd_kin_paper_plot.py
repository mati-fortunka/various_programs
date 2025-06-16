import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os

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

def plot_combined_smoothed(alpha_path, gamma_path, zeta_path,
                            alpha_dead=30, gamma_dead=30, zeta_dead=30,
                            smooth_method='savitzky_golay',
                            window_size=21, polyorder=3, x_lim=None):

    file_paths = [alpha_path, gamma_path, zeta_path]
    dead_times = [alpha_dead, gamma_dead, zeta_dead]
    labels = ["alpha", "gamma", "zeta"]
    colors = ["#75053b", "#136308", "#0721a6"]

    plt.figure(figsize=(6, 5))

    for path, dead_time, label, color in zip(file_paths, dead_times, labels, colors):
        df = read_data(path)
        time = df.iloc[:, 0] + dead_time
        signal = df.iloc[:, 1]

        if smooth_method == 'moving_average':
            smoothed_signal = moving_average(signal, window_size)
            smoothed_time = time[:len(smoothed_signal)]
        elif smooth_method == 'savitzky_golay':
            smoothed_signal = savgol_filter(signal, window_size, polyorder)
            smoothed_time = time
        else:
            raise ValueError("Unsupported smoothing method")

        plt.plot(smoothed_time, smoothed_signal, label=label, color=color)

    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Ellipticity at 222 nm [mdeg]', fontsize=16)
    #plt.title(f'Smoothed CD Kinetics Curves ({wavelength_label})')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    if x_lim:
        plt.xlim(*x_lim)
    output_folder = os.path.dirname(alpha_path)
    png_path = os.path.join(output_folder, "combined_smoothed_only.png")
    svg_path = os.path.join(output_folder, "combined_smoothed_only.svg")
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path, dpi=600)
    plt.close()

    print(f"Plots saved to:\n{png_path}\n{svg_path}")

# Example usage:
plot_combined_smoothed(
    "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/plots_paper/panel_C/alfa/8_3_alpha_5uM_2000s_nat_filt100026.csv",
    "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/plots_paper/panel_C/gamma/8_3_gamma_10ul_2000s_00006.csv",
    "/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/plots_paper/panel_C/zeta/8_3_zeta_10ul_2000s_00001.csv",
    alpha_dead=20,
    gamma_dead=40,
    zeta_dead=35,
    x_lim=(0, 2100)
)
