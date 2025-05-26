import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def read_data(filename):
    """Reads the kinetic data from a CSV file."""
    df = pd.read_csv(filename, skiprows=1)  # Skip the title row
    df = df.dropna(how='all', axis=1)  # Remove empty columns
    return df


def read_dead_times(folder):
    """Reads dead_time values from dead_time.txt."""
    dead_time_file = os.path.join(folder, "dead_time.txt")
    if not os.path.exists(dead_time_file):
        print("Warning: dead_time.txt not found. Using default dead_time=20 for all files.")
        return {}
    df = pd.read_csv(dead_time_file, sep='\t', header=None, names=['filename', 'dead_time'])
    return dict(zip(df['filename'], df['dead_time']))


def moving_average(data, window_size):
    """Applies a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def single_exponential(t, A, k, c):
    """Single exponential decay function."""
    return A * np.exp(-k * t) + c


def estimate_initial_k(time, intensity):
    """Estimates the initial decay rate k using the half-life method."""
    A0 = max(intensity)
    C = min(intensity)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity - half_max).argmin()
    t_half = time[half_max_index]
    k_init = 1 / t_half if t_half > 0 else 1
    return k_init


def fit_exponential(time, intensity):
    """Fits the intensity data to a single exponential function and returns parameters with errors."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    initial_guess = [A0, 0.01, C]
    try:
        popt, pcov = curve_fit(single_exponential, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential fit failed.")
        return None, None


def plot_each_data_separately(folder, smooth_method=None, window_size=5, polyorder=2):
    """Reads all CSV files in a folder and generates separate plots for each dataset."""
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    dead_times = read_dead_times(folder)
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for filename in csv_files:
        base_filename = os.path.basename(filename)
        dead_time = dead_times.get(base_filename, 20)
        df = read_data(filename)
        time = df.iloc[:, 0] + dead_time
        intensity = df.iloc[:, 1]
        max_intensity = max(intensity)
        intensity /= max_intensity  # Normalize

        plt.figure(figsize=(8, 5))
        plt.plot(time, intensity, label='Raw Data', alpha=0.6)

        if smooth_method == 'moving_average':
            smoothed = moving_average(intensity, window_size)
            time_adjusted = time[:len(smoothed)]
            plt.plot(time_adjusted, smoothed, linestyle='dashed', label='Smoothed')
        elif smooth_method == 'savitzky_golay':
            smoothed = savgol_filter(intensity, window_size, polyorder)
            plt.plot(time, smoothed, linestyle='dashed', label='Smoothed')

        params, errors = fit_exponential(time, intensity)
        if params is not None and errors is not None:
            fitted_curve = single_exponential(time, *params)
            plt.plot(time, fitted_curve, linestyle='dotted', label='Fit')
            print(f"Fit results for {base_filename}:")
            print(f"A = {params[0]:.4f} ± {errors[0]:.4f}")
            print(f"k = {params[1]:.4f} ± {errors[1]:.4f}")
            print(f"C = {params[2]:.4f} ± {errors[2]:.4f}")
            print("-" * 40)

        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Intensity (a.u.)')
        plt.title(f'Kinetic Data: {base_filename}')
        plt.legend()
        plt.grid()
        output_plot = os.path.join(folder, f"{base_filename}.png")
        plt.savefig(output_plot)
        plt.close()
        print(f"Plot saved as {output_plot}")


if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/slow_phase_test/kinetics/slits/unf1"
    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    plot_each_data_separately(folder_path, smooth_method=smooth_method, window_size=window_size, polyorder=polyorder)
