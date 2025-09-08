import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re

wavelength_label = "Wavelength 222 nm"

def read_data(filename):
    global wavelength_label

    # Read the txt file with tab separator and handle European commas
    df_full = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        decimal=",",   # interpret ',' as decimal point
        engine="python"
    )

    # Drop any completely empty columns
    df_full = df_full.dropna(how="all", axis=1)

    # Assign proper column names
    df_full.columns = ["Time", "Signal"]

    # Debug info
    print(f"✅ Data read: {len(df_full)} rows")
    print(df_full.head())

    # Handle time wrap
    time_col = df_full["Time"].values
    wrap_index = None
    for i in range(1, len(time_col)):
        if time_col[i] < time_col[i - 1] - 0.5:
            print(f"⚠️ Time wrap detected at row {i} (from {time_col[i - 1]} to {time_col[i]})")
            wrap_index = i
            break

    if wrap_index is not None:
        print(f"Detected time wrap at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

    # Remove duplicate timepoints
    df = df.drop_duplicates(subset="Time", keep="first")

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
              output_plot="out.png", dead_time=0,
              fit_type=None, fit_start=None, fit_end=None):
    global wavelength_label

    cd_signal = df.iloc[:, 1]
    raw_time = df.iloc[:, 0]
    abs_time = raw_time + dead_time  # absolute time including dead time

    if fit_start is not None and fit_end is not None:
        mask = (abs_time >= fit_start) & (abs_time <= fit_end)
        fit_time = abs_time[mask]  # fitting happens on absolute time scale
        fit_cd = cd_signal[mask]
    else:
        fit_time = abs_time
        fit_cd = cd_signal

    time = abs_time  # plotting also uses absolute time

    plt.figure(figsize=(8, 5))
    plt.plot(time, cd_signal, label='Raw CD Data', color='gray', alpha=0.6)

    if smooth_method == 'moving_average':
        smoothed = moving_average(cd_signal, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f'Moving Avg (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(cd_signal, window_size, polyorder)
        plt.plot(time, smoothed, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    if fit_type == "two_fits":
        fit_type = ['exponential', 'double_exponential']
    elif isinstance(fit_type, str):
        fit_type = [fit_type]

    if fit_type:
        for fit in fit_type:
            if fit == 'exponential':
                params, errors = fit_exponential(fit_time, fit_cd)
                if params is not None:
                    plt.plot(fit_time, exponential(fit_time, *params), label='Exponential Fit', color='green')
                    print("Exponential fit parameters:")
                    print(f"  A = {params[0]:.7f} ± {errors[0]:.7f}")
                    print(f"  k = {params[1]:.7f} ± {errors[1]:.7f}")
                    print(f"  c = {params[2]:.7f} ± {errors[2]:.7f}")
                    perc_error = (errors[1] / params[1]) * 100
                    print(f"  % error in k = {perc_error:.2f}%")
                    t_half = np.log(2) / params[1]
                    print(f"  t₁/₂ = {t_half:.2f} s")


            elif fit == 'exponential_drift':
                params, errors = fit_exponential_with_drift(fit_time, fit_cd)
                if params is not None:
                    plt.plot(fit_time, single_exponential_with_drift(fit_time, *params), label='Exp + Drift Fit', color='orange')
                    print("Exponential + drift fit parameters:")
                    print(f"  A = {params[0]:.7f} ± {errors[0]:.7f}")
                    print(f"  k = {params[1]:.7f} ± {errors[1]:.7f}")
                    print(f"  c = {params[2]:.7f} ± {errors[2]:.7f}")
                    print(f"  m = {params[3]:.7f} ± {errors[3]:.7f}")
                    t_half = np.log(2) / params[1]
                    perc_error = (errors[1] / params[1]) * 100
                    print(f"  % error in k = {perc_error:.2f}%")
                    print(f"  t₁/₂ = {t_half:.2f} s")

            elif fit == 'double_exponential':
                params, errors = fit_double_exponential(fit_time, fit_cd)
                if params is not None:
                    plt.plot(fit_time, double_exponential(fit_time, *params), label='Double Exp Fit', color='brown')
                    print("Double exponential fit parameters:")
                    print(f"  A1 = {params[0]:.7f} ± {errors[0]:.7f}")
                    print(f"  k1 = {params[1]:.7f} ± {errors[1]:.7f}")
                    print(f"  A2 = {params[2]:.7f} ± {errors[2]:.7f}")
                    print(f"  k2 = {params[3]:.7f} ± {errors[3]:.7f}")
                    print(f"   c = {params[4]:.7f} ± {errors[4]:.7f}")
                    t_half_k1 = np.log(2) / params[1]
                    t_half_k2 = np.log(2) / params[3]
                    perc_error_k1 = (errors[1] / params[1]) * 100
                    perc_error_k2 = (errors[3] / params[3]) * 100
                    print(f"  t₁/₂ (k1) = {t_half_k1:.2f} s | % error = {perc_error_k1:.2f}%")
                    print(f"  t₁/₂ (k2) = {t_half_k2:.2f} s | % error = {perc_error_k2:.2f}%")

            elif fit == 'linear':
                slope, intercept = fit_linear(fit_time, fit_cd)
                plt.plot(fit_time, slope * fit_time + intercept, label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}', color='purple')
                print("Linear fit parameters:")
                print(f"  slope = {slope:.7f}")
                print(f"  intercept = {intercept:.7f}")

    plt.xlabel('Time (s)')
    plt.ylabel('Ellipticity (mdeg)')
    plt.title(f'CD Kinetics Over Time\n({wavelength_label})')
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
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/CD_IIMCB/30.07.25/kin2 test2.txt"
    df = read_data(filename)

    smooth_method = "savitzky_golay"
    window_size = 5
    polyorder = 3
    dead_time = 0
    out = filename[:-4] + "_fit.png"
    print (filename.split('/')[-1])
    plot_data(df,
              smooth_method=smooth_method,
              window_size=window_size,
              polyorder=polyorder,
              output_plot=out,
              dead_time=dead_time,
              fit_type=None,
              fit_start=0,
              fit_end=2000)
