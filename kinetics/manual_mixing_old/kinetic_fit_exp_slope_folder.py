import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def read_data(filename):
    """Reads the kinetic data from a CSV file."""
    df = pd.read_csv(filename, skiprows=1)  # Skip the title row
    df = df.dropna(how='all', axis=1)  # Remove empty columns
    return df

def moving_average(data, window_size):
    """Applies a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def single_exponential_linear_drift(t, A, k, b, c):
    """Single exponential decay with linear drift."""
    return A * np.exp(-k * t) + b * t + c

def estimate_initial_k(time, intensity):
    """Estimates the initial decay rate k using the half-life method."""
    A0 = max(intensity)
    C = min(intensity)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity - half_max).argmin()
    t_half = time[half_max_index]
    return 1 / t_half if t_half > 0 else 1

def fit_exponential(time, intensity):
    """Fits the intensity data to a single exponential function with linear drift."""
    A0 = max(intensity) - min(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    b0 = (intensity.iat[-1] - intensity[0]) / (time.iat[-1] - time[0])
    initial_guess = [-0.1, 0.01, b0, C]
    try:
        popt, pcov = curve_fit(single_exponential_linear_drift, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))  # Compute standard errors
        return popt, perr
    except RuntimeError:
        return None, None

def plot_data(directory, smooth_method=None, window_size=5, polyorder=2, output_plot="overlay.png", dead_time=20):
    """Reads and overlays kinetic data from all CSV files in a directory."""
    plt.figure(figsize=(8, 5))
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = read_data(filepath)
            time = df.iloc[:, 0] + dead_time
            intensity = df.iloc[:, 1]
            max_intensity = max(intensity)
            intensity /= max_intensity  # Normalize
            plt.plot(time, intensity, label=f'{filename}', alpha=0.6)
            if smooth_method == 'moving_average':
                smoothed = moving_average(intensity, window_size)
                plt.plot(time[:len(smoothed)], smoothed, label=f'{filename} MA', linestyle='--')
            elif smooth_method == 'savitzky_golay':
                smoothed = savgol_filter(intensity, window_size, polyorder)
                plt.plot(time, smoothed, label=f'{filename} SG', linestyle='--')
            params, errors = fit_exponential(time, intensity)
            if params is not None:
                plt.plot(time, single_exponential_linear_drift(time, *params))
                print(f"Fitting parameters for {filename}:")
                print(f"A = {params[0]:.4f} ± {errors[0]:.4f}")
                print(f"k = {params[1]:.4f} ± {errors[1]:.4f}")
                print(f"b = {params[2]:.4f} ± {errors[2]:.4f}")
                print(f"c = {params[3]:.4f} ± {errors[3]:.4f}\n")
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Intensity (a.u.)')
    plt.title('Kinetic Data Overlay (Normalized)')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/slow_phase_test/kinetics/slits/ref"
    smooth_method = None
    window_size = 25
    polyorder = 3
    dead_time = 30
    output_file = os.path.join(folder_path, "overlay_exp_slope.png")
    plot_data(folder_path, smooth_method=smooth_method, window_size=window_size, polyorder=polyorder,
              output_plot=output_file, dead_time=dead_time)
