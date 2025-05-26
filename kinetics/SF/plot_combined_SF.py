import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
    exponent = np.clip(-k * t, -700, 700)  # np.exp() safe range
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
        'initial_guess': lambda t, v: [ (v[-1] - v[0]) / (t[-1] - t[0]), v[0] ],
        'param_names': ['k', 'b']
    },
    'double_exponential': {
        'func': double_exponential_model,
        'initial_guess': lambda t, v: [ (v[0] - v[-1]) / 2, 1.0, (v[0] - v[-1]) / 2, 0.1, v[-1]],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'c']
    },
    'exponential_with_linear': {
        'func': exponential_with_linear_model,
        'initial_guess': lambda t, v: [(v[0] - v[-1]), 0.01, 0.001, v[-1]],  # Smaller k and k2
        'param_names': ['a', 'k', 'k2', 'c']
    },
}

# --- Main Function ---

def plot_all_csvs_on_one_plot_with_fit(folder_path, fit_start=0.5, fit_end=10.0, model='exponential', init_guess = None):
    # Handle None or string 'none'
    fit_enabled = model is not None and str(model).lower() != 'none'

    if fit_enabled and model not in model_registry:
        print(f"Model '{model}' not recognized. Available models: {list(model_registry.keys())}")
        return

    if fit_enabled:
        model_info = model_registry[model]
        fit_func = model_info['func']
        initial_guess_func = model_info['initial_guess']
        param_names = model_info['param_names']
    else:
        model = 'none'
        param_names = []

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    plt.figure(figsize=(12, 6))
    has_data = False
    fit_results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and not filename.startswith('fit_parameters'):
            file_path = os.path.join(folder_path, filename)

            try:
                df_full = pd.read_csv(file_path, header=None, usecols=[0, 1])
                df_full.columns = ['time', 'voltage']

                # --- Wrap detection ---
                wrap_index = None
                time_col = df_full['time'].values
                for i in range(1, len(time_col)):
                    if time_col[i] <= time_col[i - 1]:
                        wrap_index = i
                        break

                if wrap_index is not None:
                    print(f"[{filename}] Detected time wrap at row {wrap_index}. Truncating data.")
                    df = df_full.iloc[:wrap_index]
                else:
                    df = df_full

                # --- Clean data ---
                df = df.drop_duplicates(subset='time', keep='first')
                df = df[df['voltage'] < 20].reset_index(drop=True)

                if len(df) < 2:
                    print(f"[{filename}] Not enough data to plot.")
                    continue

                # --- Plot raw data ---
                plt.plot(df['time'], df['voltage'], label=filename)
                has_data = True

                # --- Skip fitting if not enabled ---
                if not fit_enabled:
                    continue

                # --- Fit selected range ---
                fit_df = df[(df['time'] >= fit_start) & (df['time'] <= fit_end)]
                if len(fit_df) < len(param_names):
                    print(f"[{filename}] Not enough points in fitting range ({fit_start}s–{fit_end}s).")
                    continue

                t_fit = fit_df['time'].values
                v_fit = fit_df['voltage'].values

                if init_guess is None:
                    initial_guess = initial_guess_func(t_fit, v_fit)
                else:
                    initial_guess = init_guess
                params, covariance = curve_fit(fit_func, t_fit, v_fit, p0=initial_guess, maxfev=10000)
                errors = np.sqrt(np.diag(covariance))

                print(f"\n{filename} ({model} fit):")
                result = {'filename': filename}
                for name, val, err in zip(param_names, params, errors):
                    print(f"  {name} = {val:.6f} ± {err:.6f}")
                    result[name] = val
                    result[f"{name}_err"] = err
                fit_results.append(result)

                # --- Plot fit ---
                t_smooth = np.linspace(t_fit[0], t_fit[-1], 10000)
                v_smooth = fit_func(t_smooth, *params)
                plt.plot(t_smooth, v_smooth, linestyle='--', label=f"{filename} ({model} fit)")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if has_data:
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        title = f"Voltage vs Time"
        if fit_enabled:
            title += f" with {model.capitalize()} Fit ({fit_start}s–{fit_end}s)"
        plt.title(title)
        plt.legend()
        plt.grid()

        suffix = model if fit_enabled else "nofit"
        output_file = os.path.join(folder_path, f'combined_plot_{suffix}.png')
        plt.savefig(output_file)
        plt.show()
        print(f"\nSaved plot: {output_file}")
    else:
        print("No valid data found to plot.")

    if fit_enabled and fit_results:
        df_params = pd.DataFrame(fit_results)
        csv_path = os.path.join(folder_path, f'fit_parameters_{model}.csv')
        df_params.to_csv(csv_path, index=False)
        print(f"Saved fit parameters to: {csv_path}")

        for param in ['k', 't_half']:
            if param in param_names:
                vals = [res[param] for res in fit_results if param in res]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals, ddof=1)
                    print(f"\nParameter '{param}' across all fits:")
                    print(f"  Mean {param} = {mean:.5f}")
                    print(f"  Std. Dev. = {std:.5f}")
        # --- Compute mean, std, and half-life for kinetic params ---
        for kinetic_param in ['k', 'k1', 'k2']:
            if kinetic_param in param_names:
                values = [res[kinetic_param] for res in fit_results if kinetic_param in res]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    t_half = np.log(2) / mean_val if mean_val != 0 else np.nan

                    print(f"\nParameter '{kinetic_param}' across all fits:")
                    print(f"  Mean {kinetic_param} = {mean_val:.5f}")
                    print(f"  Std. Dev. = {std_val:.5f}")
                    print(f"  t_half (from mean {kinetic_param}) = {t_half:.5f}")


# --- Example Usage ---

plot_all_csvs_on_one_plot_with_fit(
    "/home/matifortunka/Documents/JS/data_Cambridge/8_3/A/kinetics/SF/6M/1000s",
    fit_start=0.08,
    fit_end=50,
#    init_guess = [1, 8,1,0.1, 7],
    model= "double_exponential" # Options: "sigmoid", "linear", "exponential", "double_exponential", "exponential_with_linear"
)
