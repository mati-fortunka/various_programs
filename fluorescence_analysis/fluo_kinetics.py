import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re

def read_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            print(f"Header detected: '{first_line}'")
            if "Time (s)" in first_line:
                print("\u2705 Time units confirmed as seconds (s).")
            else:
                print("\u26a0\ufe0f Warning: Time units not clearly specified as seconds (s).")
            skiprows = 1
        else:
            print("\u26a0\ufe0f No header detected or malformed header. Assuming data starts immediately.")
            skiprows = 0

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
    df_full = df_full.dropna(how='all', axis=1)

    time_col = df_full.iloc[:, 0]
    initial_time = time_col.iloc[0]

    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()
    if wrap_index > 1:
        print(f"Detected secondary block starting at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

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

def plot_data(df, smooth_method=None, window_size=5, polyorder=2,
              output_plot="out.png", dead_time=20,
              fit_type='exponential', fit_start=None, fit_end=None):
    time = df.iloc[:, 0] + dead_time
    intensity = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(time, intensity, label='Raw Data', color='gray', alpha=0.6)

    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f'Moving Avg (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    if fit_start is not None and fit_end is not None:
        mask = (time >= fit_start) & (time <= fit_end)
        fit_time = time[mask]
        fit_intensity = intensity[mask]
    else:
        fit_time = time
        fit_intensity = intensity

    if fit_type == 'exponential':
        params, errors = fit_exponential(fit_time, fit_intensity)
        if params is not None:
            plt.plot(fit_time, exponential(fit_time, *params), label='Exponential Fit', color='green')
            print("Exponential fit parameters:")
            print(f"  A = {params[0]:.5f} ± {errors[0]:.5f}")
            print(f"  k = {params[1]:.5f} ± {errors[1]:.5f}")
            print(f"  c = {params[2]:.5f} ± {errors[2]:.5f}")

    elif fit_type == 'exponential_drift':
        params, errors = fit_exponential_with_drift(fit_time, fit_intensity)
        if params is not None:
            plt.plot(fit_time, single_exponential_with_drift(fit_time, *params), label='Exp + Drift Fit', color='orange')
            print("Exponential + drift fit parameters:")
            print(f"  A = {params[0]:.5f} ± {errors[0]:.5f}")
            print(f"  k = {params[1]:.5f} ± {errors[1]:.5f}")
            print(f"  c = {params[2]:.5f} ± {errors[2]:.5f}")
            print(f"  m = {params[3]:.5f} ± {errors[3]:.5f}")

    elif fit_type == 'double_exponential':
        params, errors = fit_double_exponential(fit_time, fit_intensity)
        if params is not None:
            plt.plot(fit_time, double_exponential(fit_time, *params), label='Double Exp Fit', color='brown')
            print("Double exponential fit parameters:")
            print(f"  A1 = {params[0]:.5f} ± {errors[0]:.5f}")
            print(f"  k1 = {params[1]:.5f} ± {errors[1]:.5f}")
            print(f"  A2 = {params[2]:.5f} ± {errors[2]:.5f}")
            print(f"  k2 = {params[3]:.5f} ± {errors[3]:.5f}")
            print(f"   c = {params[4]:.5f} ± {errors[4]:.5f}")

    elif fit_type == 'linear':
        slope, intercept = fit_linear(fit_time, fit_intensity)
        plt.plot(fit_time, slope * fit_time + intercept, label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}', color='purple')
        print("Linear fit parameters:")
        print(f"  slope = {slope:.5f}")
        print(f"  intercept = {intercept:.5f}")

    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence Intensity (a.u.)')
    plt.title('Fluorescence Kinetics Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")

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

if __name__ == "__main__":
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/A/kinetics/fluo/5uM_kinetics_2h_3.csv"
    df = read_data(filename)

    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    dead_time = 30
    out = filename[:-4] + "_fit.png"

    plot_data(df,
              smooth_method=smooth_method,
              window_size=window_size,
              polyorder=polyorder,
              output_plot=out,
              dead_time=dead_time,
              fit_type='exponential',
              fit_start=0,
              fit_end=300)
