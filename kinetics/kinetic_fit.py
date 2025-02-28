import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def read_data(filename):
    """Reads the kinetic data from a CSV file."""
    df = pd.read_csv(filename, skiprows=1,sep=",")  # Skip the title row
    df = df.dropna(how='all', axis=1)  # Remove empty columns
    return df


def moving_average(data, window_size):
    """Applies a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def single_exponential(t, A, k, c):
    """Single exponential decay function."""
    return A * np.exp(-k*t ) + c


def estimate_initial_k(time, intensity):
    """Estimates the initial decay rate k using the half-life method."""
    A0 = max(intensity)  # Initial intensity
    C = min(intensity)  # Baseline

    half_max = (A0 + C) / 2  # Halfway point between max and baseline
    half_max_index = np.abs(intensity - half_max).argmin()  # Find closest value in intensity
    t_half = time[half_max_index]  # Corresponding time

    k_init = 1 / t_half if t_half > 0 else 1  # Prevent division by zero
    return k_init


def fit_exponential(time, intensity):
    """Fits the intensity data to a single exponential function."""
    A0 = max(intensity) # Initial amplitude
    C = min(intensity)  # Baseline
    k0 = estimate_initial_k(time, intensity)  # Estimate decay rate

    initial_guess = [A0, 0.01, C]
    print(f"Initial parameters: A={initial_guess[0]:.5f}, k={initial_guess[1]:.5f}, c={initial_guess[2]:.5f}'")
    try:
        popt, _ = curve_fit(single_exponential, time, intensity, p0=initial_guess)
        return popt  # Returns fitted parameters
    except RuntimeError:
        print("Exponential fit failed.")
        return None


def plot_data(df, smooth_method=None, window_size=5, polyorder=2, output_plot="out.png", dead_time=20):
    """Plots the kinetic data with optional smoothing and fitting."""
    time = df.iloc[:, 0] + dead_time  # Shift time by dead_time
    intensity = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(time, intensity, label='Raw Data', color='gray', alpha=0.6)

    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]  # Adjust time axis
        plt.plot(time_adjusted, smoothed, label=f'Moving Average (window={window_size})', color='blue')

    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    # Fit exponential curve
    params = fit_exponential(time, intensity)
    if params is not None:
        fitted_curve = single_exponential(time, *params)
        plt.plot(time, fitted_curve,
                 label=f'Exponential Fit: A={params[0]:.2f}, k={params[1]:.2f}, c={params[2]:.2f}', color='green')

    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Kinetic Data from Fluorimeter')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")
    print(f"Fitted parameters: A={params[0]:.5f}, k={params[1]:.5f}, c={params[2]:.5f}'")

if __name__ == "__main__":
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/CD_kin2.csv"  # Set the path to your CSV file here
    df = read_data(filename)

    smooth_method = 'savitzky_golay'  # Change to 'moving_average' or None for different options
    window_size = 25  # Adjust smoothing window size
    polyorder = 3  # Adjust polynomial order for Savitzky-Golay
    dead_time = 30  # Shift time by 20 seconds (or any user-specified value)
    out = filename[:-4] + "_fit.png"

    plot_data(df, smooth_method=smooth_method, window_size=window_size, polyorder=polyorder, output_plot=out,
              dead_time=dead_time)

