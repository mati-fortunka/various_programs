import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os, re, glob

# === Models ===
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
    t_half = time[np.abs(intensity - half_max).argmin()]
    return 1 / t_half if t_half > 0 else 1

# === Fit wrappers ===
def fit_exponential(time, signal):
    A0, C = max(signal), min(signal)
    guess = [A0, 0.01, C]
    try:
        popt, pcov = curve_fit(exponential, time, signal, p0=guess)
        return popt, np.sqrt(np.diag(pcov))
    except:
        return None, None

def fit_exponential_drift(time, signal):
    A0, C = max(signal), min(signal)
    guess = [A0, 0.01, C, 0.0]
    try:
        popt, pcov = curve_fit(single_exponential_with_drift, time, signal, p0=guess)
        return popt, np.sqrt(np.diag(pcov))
    except:
        return None, None

def fit_double_exponential(time, signal):
    A0, C = max(signal), min(signal)
    guess = [0.7 * A0, 0.01, 0.3 * A0, 0.001, C]
    try:
        popt, pcov = curve_fit(double_exponential, time, signal, p0=guess, maxfev=10000)
        return popt, np.sqrt(np.diag(pcov))
    except:
        return None, None

def fit_linear(time, signal):
    try:
        slope, intercept = np.polyfit(time, signal, 1)
        return [slope, intercept], [0, 0]  # no error estimated
    except:
        return None, None

# === Data and Dead Time ===
def read_data(filepath, dead_time):
    df = pd.read_csv(filepath, skiprows=1 if not filepath.endswith('_noheader.csv') else 0)
    df = df.dropna(how='all', axis=1)
    time = df.iloc[:, 0].values + dead_time
    signal = df.iloc[:, 1].values
    return time, signal

def read_dead_times(file_path):
    dead_times = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    name, val = parts
                    try:
                        dead_times[name] = int(val.rstrip('s'))
                    except ValueError:
                        pass
    return dead_times

# === Main ===
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/kinetics/CD/2000s/part1"
fit_type = 'exponential'  # or 'exponential_drift', 'double_exponential', 'linear'
default_dead_time = 30

dead_time_file = os.path.join(folder_path, "dead_times.txt")
dead_times_dict = read_dead_times(dead_time_file)

raw_curves, fitted_curves, fit_summaries, fit_params = [], [], [], []
color_map = plt.get_cmap('tab10')

for i, filepath in enumerate(sorted(glob.glob(os.path.join(folder_path, "*.csv")))):
    try:
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        dead_time = dead_times_dict.get(name, default_dead_time)

        time, signal = read_data(filepath, dead_time)

        if fit_type == 'exponential':
            popt, perr = fit_exponential(time, signal)
            model = exponential
        elif fit_type == 'exponential_drift':
            popt, perr = fit_exponential_drift(time, signal)
            model = single_exponential_with_drift
        elif fit_type == 'double_exponential':
            popt, perr = fit_double_exponential(time, signal)
            model = double_exponential
        elif fit_type == 'linear':
            popt, perr = fit_linear(time, signal)
            model = lambda t, m, b: m * t + b
        else:
            continue

        if popt is None:
            fit_summaries.append(f"{filename}: Fit failed.")
            continue

        fit_vals = model(time, *popt)
        color = color_map(i % 10)
        raw_curves.append((time, signal, name, color))
        fitted_curves.append((time, fit_vals, name, color))

        if fit_type == 'exponential':
            t_half = np.log(2) / popt[1]
            fit_summaries.append(
                f"{filename}: A={popt[0]:.6f}±{perr[0]:.6f}, k={popt[1]:.6f}±{perr[1]:.6f}, "
                f"c={popt[2]:.6f}±{perr[2]:.6f}, t_half={t_half:.2f}s")
        elif fit_type == 'exponential_drift':
            t_half = np.log(2) / popt[1]
            fit_summaries.append(
                f"{filename}: A={popt[0]:.6f}±{perr[0]:.6f}, k={popt[1]:.6f}±{perr[1]:.6f}, "
                f"c={popt[2]:.6f}±{perr[2]:.6f}, m={popt[3]:.6f}±{perr[3]:.6f}, t_half={t_half:.2f}s")
        elif fit_type == 'double_exponential':
            t1 = np.log(2) / popt[1]
            t2 = np.log(2) / popt[3]
            fit_summaries.append(
                f"{filename}: A1={popt[0]:.6f}±{perr[0]:.6f}, k1={popt[1]:.6f}±{perr[1]:.6f}, "
                f"A2={popt[2]:.6f}±{perr[2]:.6f}, k2={popt[3]:.6f}±{perr[3]:.6f}, "
                f"c={popt[4]:.6f}±{perr[4]:.6f}, t_half1={t1:.2f}s, t_half2={t2:.2f}s")
        elif fit_type == 'linear':
            fit_summaries.append(f"{filename}: slope={popt[0]:.6f}, intercept={popt[1]:.6f}")

        fit_params.append(popt)

    except Exception as e:
        fit_summaries.append(f"{filename}: Failed to process - {str(e)}")

# === Save fitting results ===
fit_summaries = list(dict.fromkeys(fit_summaries))
with open(os.path.join(folder_path, "fitting_results.txt"), "w") as f:
    f.write("\n".join(fit_summaries))

    if fit_params:
        param_array = np.array(fit_params)
        means = np.mean(param_array, axis=0)
        stds = np.std(param_array, axis=0)
        f.write("\n\nSummary of Fitted Parameters:\n")
        param_labels = {
            'exponential': ['A', 'k', 'c'],
            'exponential_drift': ['A', 'k', 'c', 'm'],
            'double_exponential': ['A1', 'k1', 'A2', 'k2', 'c'],
            'linear': ['slope', 'intercept']
        }.get(fit_type, [f"param{i}" for i in range(len(means))])
        for label, mean, std in zip(param_labels, means, stds):
            f.write(f"{label}: mean={mean:.6f}, std={std:.6f}\n")

# === Plot ===
plt.figure(figsize=(10, 6))
for (t_raw, y_raw, label, color), (_, y_fit, _, _) in zip(raw_curves, fitted_curves):
    plt.plot(t_raw, y_raw, label=f"{label} (Raw)", alpha=0.5, color=color)
    plt.plot(t_raw, y_fit, label=f"{label} (Fit)", linestyle="--", color=color)

plt.xlabel("Time (s)")
plt.ylabel("Ellipticity (mdeg)")
plt.title("Combined Raw and Fitted CD Kinetics Curves")
plt.legend(fontsize='small', loc='best')
plt.grid(True)
plt.tight_layout()
#plt.savefig(os.path.join(folder_path, "combined_raw_fitted_plot.png"))
plt.show()






