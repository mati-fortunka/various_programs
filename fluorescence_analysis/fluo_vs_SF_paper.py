import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import re

# =========================================================
# === STOPPED-FLOW MODELS (from code 1)
# =========================================================

def exponential_model(t, a, k, c):
    return a * np.exp(-k * t) + c

def sigmoid_model(t, y0, a, k, t_half):
    return y0 + a / (1 + np.exp(-k * (t - t_half)))

def linear_model(t, k, b):
    return k * t + b

def double_exponential_model(t, a1, k1, a2, k2, c):
    return a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t) + c

def exponential_with_linear_model(t, a, k, k2, c):
    exponent = np.clip(-k * t, -700, 700)
    return a * np.exp(exponent) + k2 * t + c

model_registry = {
    'exponential': {'func': exponential_model, 'initial_guess': lambda t, v: [v[0] - v[-1], 1.0, v[-1]]},
    'sigmoid': {'func': sigmoid_model, 'initial_guess': lambda t, v: [v.min(), v.max() - v.min(), 1.0, (t[0] + t[-1]) / 2]},
    'linear': {'func': linear_model, 'initial_guess': lambda t, v: [(v[-1] - v[0]) / (t[-1] - t[0]), v[0]]},
    'double_exponential': {'func': double_exponential_model, 'initial_guess': lambda t, v: [(v[0] - v[-1]) / 2, 1.0, (v[0] - v[-1]) / 2, 0.1, v[-1]]},
    'exponential_with_linear': {'func': exponential_with_linear_model, 'initial_guess': lambda t, v: [(v[0] - v[-1]), 0.01, 0.001, v[-1]]},
}

# =========================================================
# === FLUORIMETRY HELPERS (from code 2)
# =========================================================

def read_fluorimetry_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            skiprows = 1
        else:
            skiprows = 0
    skiprows = 2  # enforce in 6_3 fluo files!

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",").dropna(how='all', axis=1)
    time_col = df_full.iloc[:, 0]
    initial_time = time_col.iloc[0]
    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()

    if wrap_index > 1:
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full
    return df

# =========================================================
# === MAIN COMBINED PLOT
# =========================================================

def plot_fluorimetry_and_stoppedflow(
    fluorimetry_file,
    stoppedflow_file,
    smooth=False,
    window_length=15,
    polyorder=3,
    model=None,
    fit_start=None,
    fit_end=None,
    x_limits=None
):
    # ------------------------
    # Read Fluorimetry Data
    # ------------------------
    fluor_df = read_fluorimetry_data(fluorimetry_file)
    time_f = fluor_df.iloc[:, 0].values
    intensity = fluor_df.iloc[:, 1].values

    # ------------------------
    # Read Stopped-Flow Data
    # ------------------------
    sf_df = pd.read_csv(stoppedflow_file, header=None, usecols=[0, 1])
    sf_df.columns = ['time', 'voltage']

    wrap_index = next((i for i in range(1, len(sf_df)) if sf_df['time'][i] <= sf_df['time'][i - 1]), None)
    if wrap_index:
        sf_df = sf_df.iloc[:wrap_index]

    sf_df = sf_df.drop_duplicates(subset='time')
    sf_df = sf_df[sf_df['voltage'] < 20].reset_index(drop=True)

    time_s = sf_df['time'].values
    voltage = sf_df['voltage'].values

    if smooth:
        if window_length % 2 == 0:
            window_length += 1
        if len(voltage) >= window_length:
            voltage = savgol_filter(voltage, window_length=window_length, polyorder=polyorder)

    # ------------------------
    # Plotting
    # ------------------------
    fig, ax1 = plt.subplots(figsize=(6, 5))

    # Left axis: fluorimetry
    ax1.plot(time_f, intensity, label="Fluorimetry", color="#75053b")
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=16, color="#75053b")
    ax1.tick_params(axis='y', labelcolor="#75053b")
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    # Right axis: stopped-flow
    ax2 = ax1.twinx()
    ax2.plot(time_s, voltage, label="Stopped-Flow", color="#3D48A4")
    ax2.set_ylabel("Voltage (V)", fontsize=16, color="#3D48A4")
    ax2.tick_params(axis='y', labelcolor="#3D48A4")
    ax2.tick_params(axis='y', labelsize=15)

    # Optional fitting for stopped-flow
    if model and model in model_registry and fit_start is not None and fit_end is not None:
        model_info = model_registry[model]
        fit_func = model_info['func']
        guess_func = model_info['initial_guess']
        fit_df = sf_df[(sf_df['time'] >= fit_start) & (sf_df['time'] <= fit_end)]
        if len(fit_df) >= 3:
            t_fit = fit_df['time'].values
            v_fit = fit_df['voltage'].values
            initial_guess = guess_func(t_fit, v_fit)
            params, _ = curve_fit(fit_func, t_fit, v_fit, p0=initial_guess, maxfev=10000)
            t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
            v_smooth = fit_func(t_smooth, *params)
            ax2.plot(t_smooth, v_smooth, linestyle="--", color="#3D48A4", label=f"Stopped-Flow ({model}) fit")

    # Final cosmetics
    if x_limits:
        ax1.set_xlim(*x_limits)

    fig.tight_layout()
    fin_folder = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/"
    plt.savefig(f"{fin_folder}/fluo_vs_SF.svg", format='svg', dpi=600, bbox_inches='tight')
    plt.savefig(f"{fin_folder}/fluo_vs_SF.png", format='png', dpi=600, bbox_inches='tight')

    plt.show()


# =========================================================
# === Example usage
# =========================================================
if __name__ == "__main__":
    plot_fluorimetry_and_stoppedflow(
        fluorimetry_file="/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/fluo/double_exp/63_2h_3.csv",
        stoppedflow_file="/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv",
        smooth=True,
        window_length=15,
        polyorder=3,
        model=None,         # e.g. "exponential"
        fit_start=0,
        fit_end=1,
        x_limits=(0, 2000)
    )
