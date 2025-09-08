import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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

# --- Main Plot Function ---

def plot_three_traces_with_optional_fit_and_smoothing(
    file_paths,
    model=None,
    fit_start=0.5,
    fit_end=10.0,
    smooth=True,
    window_lengths=None,
    polyorders=None,
    default_polyorder=3,
    x_limits=None,
    y_limits=None,
):

    # labels = ["3.5 M", "4 M", "zeta"]
    # colors = ["#FB7305", "#3D48A4"]
    # colors = ["#75053b", "#136308", "#0721a6"]

    labels = ["4 M", "4.5 M", "5 M"]
    colors = ["#3D48A4", "#136308", '#FB7305']

    if window_lengths is None:
        window_lengths = [15, 15, 15]  # default same for all
    if polyorders is None:
        polyorders = [default_polyorder] * len(file_paths)

    plt.figure(figsize=(6, 5))

    for idx, (path, label, color) in enumerate(zip(file_paths, labels, colors)):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        try:
            df = pd.read_csv(path, header=None, usecols=[0, 1])
            df.columns = ['time', 'voltage']

            wrap_index = next((i for i in range(1, len(df)) if df['time'][i] <= df['time'][i - 1]), None)
            if wrap_index:
                df = df.iloc[:wrap_index]

            df = df.drop_duplicates(subset='time')
            df = df[df['voltage'] < 20].reset_index(drop=True)

            if path == "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4.5M/2000s/6_3_2uM_2000s00021.csv":
                df['voltage'] -= 1.2
            elif path == "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv":
                df['voltage'] += 1

            time = df['time'].values
            voltage = df['voltage'].values

            if smooth:
                win = window_lengths[idx]
                poly = polyorders[idx]
                if win % 2 == 0:
                    win += 1  # ensure odd
                if len(voltage) >= win:
                    voltage = savgol_filter(voltage, window_length=win, polyorder=poly)
                else:
                    print(f"{label}: Not enough points for smoothing (needs at least {win}); skipping it.")

            plt.plot(time, voltage, label=label, color=color)

            if model and model in model_registry:
                model_info = model_registry[model]
                fit_func = model_info['func']
                param_names = model_info['param_names']
                guess_func = model_info['initial_guess']

                fit_df = df[(df['time'] >= fit_start) & (df['time'] <= fit_end)]
                if len(fit_df) < len(param_names):
                    print(f"{label}: Not enough points for fitting; skipping.")
                    continue

                t_fit = fit_df['time'].values
                v_fit = fit_df['voltage'].values
                initial_guess = guess_func(t_fit, v_fit)

                params, _ = curve_fit(fit_func, t_fit, v_fit, p0=initial_guess, maxfev=10000)

                t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
                v_smooth = fit_func(t_smooth, *params)
                plt.plot(t_smooth, v_smooth, linestyle='--', color=color, label=f"{label} ({model}) fit")

        except Exception as e:
            print(f"Error processing {label}: {e}")

    if x_limits:
        plt.xlim(*x_limits)
    if y_limits:
        plt.ylim(*y_limits)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Voltage (V)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()

    """
    fast
    fin_folder = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/SF"
    plt.savefig(f"{fin_folder}/fast.svg", format='svg', dpi=600, bbox_inches='tight')
    plt.savefig(f"{fin_folder}/fast.png", format='png', dpi=600, bbox_inches='tight')
    """

    fin_folder = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/SF"
    plt.savefig(f"{fin_folder}/slow_v3.svg", format='svg', dpi=600, bbox_inches='tight')
    plt.savefig(f"{fin_folder}/slow_v3.png", format='png', dpi=600, bbox_inches='tight')

    plt.show()


plot_three_traces_with_optional_fit_and_smoothing(
    file_paths=[
       "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv",
       "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4.5M/2000s/6_3_2uM_2000s00021.csv",
       "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/5M/2000s/6_3_2uM_2000s00012.csv"
    ],
    # #2nd ver
    # file_paths=[
    #    "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv",
    #    "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/4.5M/2000s/6_3_2uM_2000s00021.csv",
    #    "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/5M/2000s/6_3_2uM_2000s00012.csv"
    # ],
    model=None,          # or None to disable fitting
    fit_start=0,
    fit_end=1000,
    smooth=False,              # Toggle smoothing on/off
    window_lengths=[5, 5],   # Different for each trace
    polyorders=[3, 3],
    x_limits=(0, 2000),
    y_limits=None
)
