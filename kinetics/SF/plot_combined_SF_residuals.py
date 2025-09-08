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
    exponent = np.clip(-k * t, -700, 700)
    return a * np.exp(exponent) + k2 * t + c


def double_exponential_sigmoid_model(t, a1, k1, a2, k2, c, y0, a, k, t_half):
    return double_exponential_model(t, a1, k1, a2, k2, c) + sigmoid_model(t, y0, a, k, t_half)


def double_exponential_linear_model(t, a1, k1, a2, k2, k3, c):
    return a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t) + k3 * t + c


def triple_exponential_model(t, a1, k1, a2, k2, a3, k3, c):
    return (
            a1 * np.exp(-k1 * t)
            + a2 * np.exp(-k2 * t)
            + a3 * np.exp(-k3 * t)
            + c
    )


def triple_exponential_linear_model(t, a1, k1, a2, k2, a3, k3, k_lin, c):
    return (
            a1 * np.exp(-k1 * t)
            + a2 * np.exp(-k2 * t)
            + a3 * np.exp(-k3 * t)
            + k_lin * t
            + c
    )


def quadruple_exponential_model(t, a1, k1, a2, k2, a3, k3, a4, k4, c):
    return (
            a1 * np.exp(-k1 * t)
            + a2 * np.exp(-k2 * t)
            + a3 * np.exp(-k3 * t)
            + a4 * np.exp(-k4 * t)
            + c
    )


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
    'double_exponential_sigmoid': {
        'func': double_exponential_sigmoid_model,
        'initial_guess': lambda t, v: [
            (v[0] - v[-1]) / 2, 1.0,
            (v[0] - v[-1]) / 2, 0.1,
            v[-1],
            v.min(), v.max() - v.min(), 1.0, 1000.0
        ],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'c', 'y0', 'a', 'k', 't_half']
    },
    'double_exponential_linear': {
        'func': double_exponential_linear_model,
        'initial_guess': lambda t, v: [
            (v[0] - v[-1]) / 2, 1.0,
            (v[0] - v[-1]) / 2, 0.1,
            0.001, v[-1]
        ],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'k3', 'c']
    },
    'triple_exponential': {
        'func': triple_exponential_model,
        'initial_guess': lambda t, v: [
            (v[0] - v[-1]) / 3, 1.0,
            (v[0] - v[-1]) / 3, 0.3,
            (v[0] - v[-1]) / 3, 0.05,
            v[-1]
        ],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'a3', 'k3', 'c']
    },
    'triple_exponential_linear': {
        'func': triple_exponential_linear_model,
        'initial_guess': lambda t, v: [
            (v[0] - v[-1]) / 3, 1.0,
            (v[0] - v[-1]) / 3, 0.3,
            (v[0] - v[-1]) / 3, 0.05,
            0.001, v[-1]
        ],
        # 'initial_guess': lambda t, v: [
        #     0.06, 0.4,
        #     0.3, 0.008,
        #     1.5, 0.0013,
        #     -0.0001, v[-1]
        # ],
        # init_guess = [0.03, 0.6,0.16,0.01, 1.5,0.002, -0.0003,9],
# a1 = 0.0630748 ± 0.0069006
#   k1 = 0.3605565 ± 0.0527502
#   a2 = 0.3263099 ± 0.0083099
#   k2 = 0.0079068 ± 0.0001791
#   a3 = 1.4942256 ± 0.0024908
#   k3 = 0.0013619 ± 0.0000204
#   c = 6.1966625 ± 0.0079368
        'param_names': ['a1', 'k1', 'a2', 'k2', 'a3', 'k3', 'k_lin', 'c']
    },
    'quadruple_exponential': {
        'func': quadruple_exponential_model,
        'initial_guess': lambda t, v: [
            (v[0] - v[-1]) / 4, 1.0,
            (v[0] - v[-1]) / 4, 0.3,
            (v[0] - v[-1]) / 4, 0.1,
            (v[0] - v[-1]) / 4, 0.05,
            v[-1]
        ],
        # 'initial_guess': lambda t, v: [
        #     0.06, 0.36,
        #     0.3, 0.008,
        #     0.6, 0.001,
        #     0.8, 0.001,
        #     v[-1]
        # ],
        'param_names': ['a1', 'k1', 'a2', 'k2', 'a3', 'k3', 'a4', 'k4', 'c']
    },
}


# --- Main Function ---

def plot_all_csvs_on_one_plot_with_fit(folder_path, fit_start=0.5, fit_end=10.0, model='exponential', init_guess=None):
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

    fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    has_data = False
    fit_results = []

    color_cycle = plt.cm.tab10.colors
    color_index = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and not filename.startswith('fit_parameters'):
            file_path = os.path.join(folder_path, filename)

            try:
                df_full = pd.read_csv(file_path, header=None, usecols=[0, 1])
                df_full.columns = ['time', 'voltage']

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

                df = df.drop_duplicates(subset='time', keep='first')
                df = df[df['voltage'] < 20].reset_index(drop=True)

                if len(df) < 2:
                    print(f"[{filename}] Not enough data to plot.")
                    continue

                color = color_cycle[color_index % len(color_cycle)]
                color_index += 1

                ax_main.plot(df['time'], df['voltage'], label=filename, color=color)
                has_data = True

                if not fit_enabled:
                    continue

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

                # Check for negative initial guesses for k values and adjust if necessary
                for i, name in enumerate(param_names):
                    if name.startswith('k') and initial_guess[i] <= 0:
                        initial_guess[i] = abs(initial_guess[i]) + 1e-6  # Ensure k is positive

                params, covariance = curve_fit(fit_func, t_fit, v_fit, p0=initial_guess, maxfev=20000)
                errors = np.sqrt(np.diag(covariance))

                print(f"\n{filename} ({model} fit):")
                result = {'filename': filename}
                for name, val, err in zip(param_names, params, errors):
                    print(f"  {name} = {val:.7f} ± {err:.7f}")
                    result[name] = val
                    result[f"{name}_err"] = err

                # Handle t_halfs for all models with exponential components
                t_half_data = {}
                k_names = [name for name in param_names if name.startswith('k') and name != 'k_lin']

                if k_names:
                    print("\n  Calculated t_halfs:")

                for k_name in k_names:
                    k_val = result.get(k_name)
                    if k_val and k_val > 0:
                        t_half = np.log(2) / k_val
                        t_half_name = f't_half_{k_name}'
                        result[t_half_name] = t_half
                        t_half_data[t_half_name] = t_half
                        print(f"  {t_half_name} = {t_half:.7f}")

                # Handle amplitude ratios for multi-exponential models
                if model in ['double_exponential', 'double_exponential_linear', 'double_exponential_sigmoid']:
                    a1 = result.get('a1')
                    a2 = result.get('a2')
                    if a1 is not None and a2 is not None:
                        total_a = abs(a1) + abs(a2)
                        if total_a != 0:
                            ratio = abs(a1) / total_a
                            result['a1_ratio'] = ratio
                            print(f"  a1/(a1+a2) = {ratio:.7f}")

                if model in ['triple_exponential', 'triple_exponential_linear']:
                    a1 = result.get('a1')
                    a2 = result.get('a2')
                    a3 = result.get('a3')
                    if a1 is not None and a2 is not None and a3 is not None:
                        total_a = abs(a1) + abs(a2) + abs(a3)
                        if total_a != 0:
                            ratio1 = abs(a1) / total_a
                            ratio2 = abs(a2) / total_a
                            ratio3 = abs(a3) / total_a
                            result['a1_ratio'] = ratio1
                            result['a2_ratio'] = ratio2
                            result['a3_ratio'] = ratio3
                            print(f"  a1/(a1+a2+a3) = {ratio1:.7f}")
                            print(f"  a2/(a1+a2+a3) = {ratio2:.7f}")
                            print(f"  a3/(a1+a2+a3) = {ratio3:.7f}")

                fit_results.append(result)

                t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
                v_smooth = fit_func(t_smooth, *params)
                ax_main.plot(t_smooth, v_smooth, linestyle='--', label=f"{filename} ({model} fit)", color=color)

                # Residuals
                residuals = v_fit - fit_func(t_fit, *params)
                ax_resid.plot(t_fit, residuals, '.', color=color)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if has_data:
        ax_main.set_ylabel("Voltage (V)")
        ax_resid.set_xlabel("Time (s)")
        ax_resid.set_ylabel("Residuals")
        ax_resid.axhline(0, color='black', linestyle='--', linewidth=1)

        title = f"Voltage vs Time"
        if fit_enabled:
            title += f" with {model.capitalize()} Fit ({fit_start}s–{fit_end}s)"
        ax_main.set_title(title)

        ax_main.legend()
        ax_main.grid()
        ax_resid.grid()

        suffix = model if fit_enabled else "nofit"
        output_file = os.path.join(folder_path, f'combined_plot_{suffix}.png')
        plt.tight_layout()
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

        # Calculate and print mean/std dev for all t_halfs
        t_half_cols = [col for col in df_params.columns if 't_half_' in col]
        if t_half_cols:
            print("\nSummary statistics for t_halfs across all fits:")
            for t_half_col in t_half_cols:
                vals = df_params[t_half_col].dropna()
                if not vals.empty:
                    mean_t = np.mean(vals)
                    std_t = np.std(vals, ddof=1)
                    print(f"  {t_half_col}:")
                    print(f"    Mean = {mean_t:.7f}")
                    print(f"    Std. Dev. = {std_t:.7f}")

        # Calculate and print mean/std dev for all amplitude ratios
        ratio_cols = [col for col in df_params.columns if 'ratio' in col]
        if ratio_cols:
            print("\nSummary statistics for amplitude ratios across all fits:")
            for ratio_col in ratio_cols:
                vals = df_params[ratio_col].dropna()
                if not vals.empty:
                    mean_ratio = np.mean(vals)
                    std_ratio = np.std(vals, ddof=1)
                    print(f"  {ratio_col}:")
                    print(f"    Mean = {mean_ratio:.7f}")
                    print(f"    Std. Dev. = {std_ratio:.7f}")


# --- Example Usage ---

plot_all_csvs_on_one_plot_with_fit(
    "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/unfolding/5M/2000s",
    fit_start=0.4,
    fit_end=900,
    # init_guess = [0.03, 0.6,0.16,0.01, 1.5,0.002, -0.0003,9],
    model='triple_exponential'
    # Options: "sigmoid", "linear", "exponential", "double_exponential", "exponential_with_linear", "triple_exponential_linear"
)