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
    """Fits the intensity data to a single exponential function with linear drift and prints fitting errors."""
    A0 = max(intensity) - min(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    b0 = (intensity.iat[-1] - intensity[0]) / (time.iat[-1] - time[0])  # Estimate linear drift
    initial_guess = [A0, k0, b0, C]
    initial_guess = [-0.1, 0.01, b0, C]

    print(
        f"Initial parameters: A={initial_guess[0]:.5f}, k={initial_guess[1]:.5f}, b={initial_guess[2]:.5f}, c={initial_guess[3]:.5f}")

    try:
        popt, pcov = curve_fit(single_exponential_linear_drift, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))  # Standard deviation errors of parameters

        print("Fitting errors:")
        print(f"A error: {perr[0]:.5f}")
        print(f"k error: {perr[1]:.5f}")
        print(f"b error: {perr[2]:.5f}")
        print(f"c error: {perr[3]:.5f}")

        return popt, perr
    except RuntimeError:
        print("Exponential fit failed.")
        return None, None


def plot_data(df, smooth_method=None, window_size=5, polyorder=2, output_plot="out.png", dead_time=20):
    """Plots the kinetic data with optional smoothing and fitting."""
    time = df.iloc[:, 0] + dead_time
    intensity = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(time, intensity, label='Raw Data', color='gray', alpha=0.6)

    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f'Moving Average (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    # Fit exponential curve with linear drift
    params, errors = fit_exponential(time, intensity)
    if params is not None:
        fitted_curve = single_exponential_linear_drift(time, *params)
        plt.plot(time, fitted_curve,
                 label=f'Fit: A={params[0]:.2f}, k={params[1]:.2f}, b={params[2]:.2f}, c={params[3]:.2f}',
                 color='green')

    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Kinetic Data with Exponential + Linear Drift Fit')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")
    if params is not None:
        print(f"Fitted parameters: A={params[0]:.5f} ± {errors[0]:.5f}, k={params[1]:.5f} ± {errors[1]:.5f}, "
              f"b={params[2]:.5f} ± {errors[2]:.5f}, c={params[3]:.5f} ± {errors[3]:.5f}")


if __name__ == "__main__":
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/slow_phase_test/kinetics/slits/Tm1570_5M_2uM_30min_emslit20.csv"
    df = read_data(filename)
    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    dead_time = 30
    out = filename[:-4] + "_exp_slope.png"

    plot_data(df, smooth_method=smooth_method, window_size=window_size, polyorder=polyorder, output_plot=out,
              dead_time=dead_time)