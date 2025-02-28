import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def read_data(filename):
    """Reads the kinetic data from a CSV file."""
    df = pd.read_csv(filename, skiprows=1)  # Skip the title row
    df = df.dropna(how='all', axis=1)  # Remove empty columns
    return df


def moving_average(data, window_size):
    """Applies a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_data(df, smooth_method=None, window_size=5, polyorder=2, output_plot="out.png"):
    """Plots the kinetic data with optional smoothing."""
    time = df.iloc[:, 0]
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

    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Kinetic Data from Fluorimeter')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")

if __name__ == "__main__":
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/slow_phase_test/kinetics/Tm1570_5M_1800s_2uM.csv"  # Set the path to your CSV file here
    df = read_data(filename)

    smooth_method = 'savitzky_golay'  # Change to 'moving_average' or None for different options
    window_size = 25  # Adjust smoothing window size
    polyorder = 3  # Adjust polynomial order for Savitzky-Golay
    out = filename[:-4]+".png"

    plot_data(df, smooth_method=smooth_method, window_size=window_size, polyorder=polyorder, output_plot=out)
