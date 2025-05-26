import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def extract_urea_concentration(filename):
    """Extract urea concentration from filename, ensuring robustness."""
    match = re.search(r'_(\d{1,2})Murea', filename)
    if match:
        try:
            concentration = float(match.group(1))
            return concentration / 10 if len(match.group(1)) == 2 else concentration
        except ValueError:
            return None
    return None


def single_exponential(t, A, k, c):
    return A * np.exp(-k * t) + c


def single_exponential_linear_drift(t, A, k, b, c):
    return A * np.exp(-k * t) + b * t + c


def double_exponential(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c

def linear_fit(x, y):
    coeffs = np.polyfit(x, y, 1)
    print(f"Linear fit parameters: Slope={coeffs[0]}, Intercept={coeffs[1]}")
    return coeffs

def estimate_initial_k(time, intensity):
    if len(time) < 2:
        return 1  # Fallback for insufficient data

    A0, C = max(intensity), min(intensity)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity.values - half_max).argmin()
    t_half = time.iloc[half_max_index]
    return 1 / t_half if t_half > 0 else 1


def estimate_initial_double_k(time, intensity):
    if len(time) < 10:
        return 1, 0.1  # Fallback for very short time series

    split_index = max(1, len(time) // 5)  # Adjust split to capture more initial decay

    early_time = time.iloc[:split_index]
    early_intensity = intensity.iloc[:split_index]
    late_time = time.iloc[split_index:]
    late_intensity = intensity.iloc[split_index:]

    k1 = estimate_initial_k(early_time, early_intensity)
    k2 = estimate_initial_k(late_time, late_intensity) / 2  # Reduce second k to ensure different decay rates

    return max(k1, k2 * 1.5), min(k1, k2 / 1.5)  # Ensure k1 > k2


def fit_data(x, y, fit_type="single_exponential"):
    """Fit data to the specified function, handling errors gracefully."""
    try:
        if fit_type == "single_exponential":
            initial_guess = [max(y), estimate_initial_k(x, y), min(y)]
            param_names = ["A", "k", "c"]
            popt, pcov = curve_fit(single_exponential, x, y, p0=initial_guess, maxfev=5000)
        elif fit_type == "single_exponential_linear_drift":
            b0 = (y.iloc[-1] - y.iloc[0]) / (x.iloc[-1] - x.iloc[0])
            initial_guess = [max(y) - min(y), estimate_initial_k(x, y), b0, min(y)]
            param_names = ["A", "k", "b", "c"]
            popt, pcov = curve_fit(single_exponential_linear_drift, x, y, p0=initial_guess, maxfev=5000)
        elif fit_type == "double_exponential":
            k1, k2 = estimate_initial_double_k(x, y)
            initial_guess = [max(y) / 2, k1, max(y) / 2, k2, min(y)]
            param_names = ["A1", "k1", "A2", "k2", "c"]
            popt, pcov = curve_fit(double_exponential, x, y, p0=initial_guess, maxfev=5000)
        else:
            raise ValueError("Unsupported fit type")

        errors = np.sqrt(np.diag(pcov))
        return popt, errors, param_names
    except Exception as e:
        print(f"Fit error: {e}")
        return None, None, None


def process_files(folder_path, start_time, end_time, fit_type="single_exponential"):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            concentration = extract_urea_concentration(filename)
            if concentration is None:
                continue

            try:
                df = pd.read_csv(os.path.join(folder_path, filename), header=None, names=['time', 'voltage'])
                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

                popt, errors, param_names = fit_data(df['time'], df['voltage'], fit_type)
                if popt is not None:
                    results.setdefault(concentration, []).append((popt, errors, param_names))
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    summary = {conc: (np.mean([d[0] for d in data], axis=0), np.std([d[0] for d in data], axis=0), data[0][2])
               for conc, data in results.items()}
    return summary


def plot_results(summary, param_name="k", second_param_name="A"):
    concentrations = sorted(summary.keys())
    param_index = summary[concentrations[0]][2].index(param_name)
    second_param_index = summary[concentrations[0]][2].index(second_param_name)

    def plot_parameter(param_idx,param_name):
        values = [summary[c][0][param_idx] for c in concentrations]
        errors = [summary[c][1][param_idx] for c in concentrations]
        if param_name in ["k", "k1", "k2"]:
            values = np.log(values)

        linear_fit(concentrations, values)

        plt.errorbar(concentrations, values, yerr=errors, fmt='o', label='Data')
        plt.xlabel("Urea Concentration (M)")
        plt.ylabel(f"ln({param_name})" if param_name in ["k", "k1", "k2"] else param_name)
        plt.title(f"{param_name} vs Urea Concentration")
        plt.legend()
        plt.grid()
        plt.show()

    plot_parameter(param_index, param_name)
    plot_parameter(second_param_index, second_param_name)


# Example usage
summary = process_files("/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Tm1570/SF/to_fit/unf/2uM",
                        start_time=0.1, end_time=1000, fit_type="single_exponential")
plot_results(summary, param_name="k", second_param_name="A")