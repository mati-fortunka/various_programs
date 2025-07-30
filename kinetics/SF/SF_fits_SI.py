import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import re

# --- Model Functions ---

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

# --- Model Registry ---

model_registry = {
    'exponential': {
        'func': exponential_model,
        'initial_guess': lambda t, v: [v[0] - v[-1], 1.0, v[-1]],
        'param_names': ['a', 'k', 'c']
    },
    'sigmoid': {
        'func': sigmoid_model,
        'initial_guess': lambda t, v: [v.min(), v.max() - v.min(), 1.0, (t[0] + t[-1]) / 2],
        'param_names': ['y0', 'a', 'k', 't_half']
    },
    'linear': {
        'func': linear_model,
        'initial_guess': lambda t, v: [(v[-1] - v[0]) / (t[-1] - t[0]), v[0]],
        'param_names': ['k', 'b']
    },
    'double_exponential': {
        'func': double_exponential_model,
        'initial_guess': lambda t, v: [(v[0] - v[-1]) / 2, 1.0, (v[0] - v[-1]) / 2, 0.1, v[-1]],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'c']
    },
    'exponential_with_linear': {
        'func': exponential_with_linear_model,
        'initial_guess': lambda t, v: [(v[0] - v[-1]), 0.01, 0.001, v[-1]],
        'param_names': ['a', 'k', 'k2', 'c']
    },
}

# --- Main Function ---

def plot_all_traces_with_fit(
    folder_path,
    model='exponential',
    fit_start=0.5,
    fit_end=10.0,
    smooth=True,
    window_length=15,
    polyorder=3,
    x_limits=None,
    y_limits=None,
):
    protein_colors = {
        'alpha': '#75053b',
        'gamma': '#136308',
        'zeta': '#0721a6'
    }

    plt.figure(figsize=(6, 5))

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)
        match = re.search(r'(alpha|gamma|zeta)', filename)
        if not match:
            print(f"Skipping file without known protein label: {filename}")
            continue
        protein = match.group(1)
        color = protein_colors.get(protein, 'gray')

        try:
            df = pd.read_csv(file_path, header=None, usecols=[0, 1])
            df.columns = ['time', 'voltage']
            wrap_index = next((i for i in range(1, len(df)) if df['time'][i] <= df['time'][i - 1]), None)
            if wrap_index:
                df = df.iloc[:wrap_index]
            df = df.drop_duplicates(subset='time')
            df = df[df['voltage'] < 20].reset_index(drop=True)

            time = df['time'].values
            voltage = df['voltage'].values

            if smooth:
                if window_length % 2 == 0:
                    window_length += 1
                if len(voltage) >= window_length:
                    voltage = savgol_filter(voltage, window_length=window_length, polyorder=polyorder)
                else:
                    print(f"{filename}: Not enough points for smoothing; skipping it.")

            plt.plot(time, voltage, label=f"{filename} ({protein})", color=color)

            if model and model in model_registry:
                model_info = model_registry[model]
                fit_func = model_info['func']
                guess_func = model_info['initial_guess']
                param_names = model_info['param_names']

                fit_df = df[(df['time'] >= fit_start) & (df['time'] <= fit_end)]
                if len(fit_df) < len(param_names):
                    print(f"{filename}: Not enough points for fitting; skipping.")
                    continue

                t_fit = fit_df['time'].values
                v_fit = fit_df['voltage'].values
                initial_guess = guess_func(t_fit, v_fit)
                params, _ = curve_fit(fit_func, t_fit, v_fit, p0=initial_guess, maxfev=10000)
                t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
                v_smooth = fit_func(t_smooth, *params)
                plt.plot(t_smooth, v_smooth, linestyle='--', color=color, label=f"{filename} fit")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if x_limits:
        plt.xlim(*x_limits)
    if y_limits:
        plt.ylim(*y_limits)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Voltage (V)", fontsize=16)
    #plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f"{folder_path}/phase_D_SI.svg", format='svg', dpi=600, bbox_inches='tight')
    plt.savefig(f"{folder_path}/phase_D_SI.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()

# Example call:
plot_all_traces_with_fit(
    folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/sigmoid_D",
    model='sigmoid',
    fit_start=1000,
    fit_end=2000,
    smooth=False,
    window_length=15,
    polyorder=3,
    x_limits=(1000, 2000),
    y_limits=None
)

# phase A
# plot_all_traces_with_fit(
#     folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/phase_A",
#     model='exponential',
#     fit_start=0.1,
#     fit_end=1,
#     smooth=False,
#     window_length=15,
#     polyorder=3,
#     x_limits=(-0.0001, 1),
#     y_limits=(5.5,10)
# )

# phase B-C
# plot_all_traces_with_fit(
#     folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/double_exp_B-C",
#     model='double_exponential',
#     fit_start=1,
#     fit_end=1050,
#     smooth=False,
#     window_length=15,
#     polyorder=3,
#     x_limits=(-4, 1050),
#     y_limits=(6.4,12)
# )

# phase D
# plot_all_traces_with_fit(
#     folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/sigmoid_D",
#     model='sigmoid',
#     fit_start=1000,
#     fit_end=2000,
#     smooth=False,
#     window_length=15,
#     polyorder=3,
#     x_limits=(1000, 2000),
#     y_limits=None
# )
