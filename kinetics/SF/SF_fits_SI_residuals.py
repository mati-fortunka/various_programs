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
        'dzeta': '#0721a6'
    }

    # --- MODIFIED: Setup figure with 2 subplots ---
    # Create a figure with 2 rows, 1 column.
    # sharex=True links the x-axes.
    # gridspec_kw controls the relative height (3:1 ratio).
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 7),  # Adjusted height for two plots
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    # ax1 is the top plot (main data)
    # ax2 is the bottom plot (residuals)
    # ---

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)
        match = re.search(r'(alpha|gamma|dzeta)', filename)
        if not match:
            print(f"Skipping file without known protein label: {filename}")
            continue
        protein = match.group(1)
        # if protein == "gamma":
        #     fit_end = 1650
        # else:
        #     fit_end = 1990

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

            # --- MODIFIED: Plot data on ax1 ---
            ax1.plot(time, voltage, label=f"{filename} ({protein})", color=color)

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

                # --- MODIFIED: Plot fit on ax1 ---
                t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
                v_smooth = fit_func(t_smooth, *params)
                ax1.plot(t_smooth, v_smooth, linestyle='--', color=color, label=f"{filename} fit")

                # --- ADDED: Calculate and plot residuals on ax2 ---
                v_predicted = fit_func(t_fit, *params)
                residuals = v_fit - v_predicted
                # Plot residuals as small points
                ax2.plot(t_fit, residuals, 'o', color=color, markersize=2, alpha=0.7)
                # ---

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # --- MODIFIED: Format axes ---

    # Set limits
    if x_limits:
        ax1.set_xlim(*x_limits)  # ax2 will inherit this due to sharex=True
    if y_limits:
        ax1.set_ylim(*y_limits)

    # Format ax1 (Top plot)
    ax1.set_ylabel("Intrinsic fluorescence (a.u.)", fontsize=16)
    # ax1.legend(fontsize=10, frameon=False) # Keep commented as in original
    ax1.tick_params(axis='y', labelsize=15)
    # x-tick labels are automatically hidden by sharex=True

    # Format ax2 (Bottom plot)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add y=0 reference line
    ax2.set_xlabel("Time (s)", fontsize=16)
    ax2.set_ylabel("Residuals (a.u.)", fontsize=16)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)  # Explicitly set x-tick size

    # Adjust vertical spacing between plots to be small
    fig.subplots_adjust(hspace=0.1)

    # ---

    plt.savefig(f"{folder_path}/phase_D_SI.svg", format='svg', dpi=600, bbox_inches='tight')
    plt.savefig(f"{folder_path}/phase_D_SI.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()


# phase A
plot_all_traces_with_fit(
    folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/phase_A",
    model='exponential',
    fit_start=0.1,
    fit_end=1,
    smooth=False,
    window_length=15,
    polyorder=3,
    x_limits=(0.1, 1),
    y_limits=(5.5, 10)
)

# phase B-C
plot_all_traces_with_fit(
    folder_path="/home/matifortunka/Documents/JS/data_Cambridge/8_3/paper/SI_plots/SF_kinetics/double_exp_B-C",
    model='double_exponential',
    fit_start=2,
    fit_end=1050,
    smooth=False,
    window_length=15,
    polyorder=3,
    x_limits=(-4, 1000),
    y_limits=(6.4, 12)
)

# phase D
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

# # Example call:
# plot_all_traces_with_fit(
#     folder_path="/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/5M/1s",
#     model='double_exponential',
#     fit_start=0.005,
#     fit_end=1,
#     smooth=False,
#     window_length=15,
#     polyorder=3,
#     x_limits=None,
#     y_limits=None
# )