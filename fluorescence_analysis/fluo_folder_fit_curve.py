# Merged analysis script for three fluorescence approaches (fluorimeter data)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

RT = 0.592  # RT constant kcal

def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))

def G_three_state_weighted(x, a_n, a_i, a_u, m1, d1, m2, d2):
    sigmoid1 = 1 / (1 + np.exp(-(m1 * (x - d1)) / RT))
    sigmoid2 = 1 / (1 + np.exp(-(m2 * (x - d2)) / RT))
    return a_n * (1 - sigmoid1) + a_i * (sigmoid1 - sigmoid2) + a_u * sigmoid2

def guess_initial_params(x, y, model="two_state"):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    if model == "two_state":
        a_n = np.mean(y_sorted[:len(x) // 3])
        a_u = np.mean(y_sorted[-len(x) // 3:])
        dy_dx = np.gradient(y_sorted, x_sorted)
        d = x_sorted[np.argmax(np.abs(dy_dx))]
        m = 5 / max(np.ptp(x), 1e-6)
        return [a_n, a_u, m, d]

    elif model == "three_state":
        n = len(x_sorted)
        third = n // 3
        a_n = np.mean(y_sorted[:third])
        a_i = np.mean(y_sorted[third:2 * third])
        a_u = np.mean(y_sorted[2 * third:])
        dy_dx = np.gradient(y_sorted, x_sorted)
        transition_indices = np.argsort(np.abs(dy_dx))[-2:]
        d1, d2 = sorted(x_sorted[transition_indices])
        m1 = m2 = 5 / max(np.ptp(x), 1e-6)
        return [a_n, a_i, a_u, m1, d1, m2, d2]
    else:
        raise ValueError("Invalid model")

def smooth_data(x, y, method, window_size, spline_smoothing_factor, poly_order):
    if method == "moving_average":
        return pd.Series(y).rolling(window=window_size, center=True).mean().to_numpy()
    elif method == "spline":
        return UnivariateSpline(x, y, s=spline_smoothing_factor)(x)
    elif method == "savitzky_golay":
        return savgol_filter(y, window_length=window_size, polyorder=poly_order)
    return y

def extract_sample_number(name):
    return int(re.findall(r'(\d+)', name)[0])

def run_ew_analysis(config):
    results = []
    concentration_file = config['concentration_file'] or os.path.join(config['folder'], 'concentrations.txt')
    conc_df = pd.read_csv(concentration_file, sep="\t")
    conc_map = conc_df.set_index("Sample_number")["den_concentration"]

    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(config['folder'], fname), header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])
            df.dropna(inplace=True)
            df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'], config['smoothing'], config['window'], config['spline_s'], config['poly'])
            baseline = np.interp(config['baseline'], df['Wavelength'], df['Smoothed']) if config['baseline'] else 0
            df['Corrected'] = df['Smoothed'] - baseline
            ew = (df['Corrected'] * df['Wavelength']).sum()
            sn = extract_sample_number(fname)
            if sn in conc_map:
                results.append((conc_map[sn], ew))

    df = pd.DataFrame(results, columns=['den_concentration', 'EW']).sort_values('den_concentration')
    fit_and_plot(df, 'EW', config, 'Equivalent Width (a.u.)')

def run_ratio_analysis(config):
    results = []
    concentration_file = config['concentration_file'] or os.path.join(config['folder'], 'concentrations.txt')
    conc_df = pd.read_csv(concentration_file, sep="\t")
    conc_map = conc_df.set_index("Sample_number")["den_concentration"]

    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(config['folder'], fname), header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])
            df.dropna(inplace=True)
            df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'], config['smoothing'], config['window'], config['spline_s'], config['poly'])
            baseline = np.interp(config['baseline'], df['Wavelength'], df['Smoothed']) if config['baseline'] else 0
            df['Corrected'] = df['Smoothed'] - baseline

            i1 = df.iloc[(df['Wavelength'] - config['wl1']).abs().argsort()[:1]]['Corrected'].values[0]
            i2 = df.iloc[(df['Wavelength'] - config['wl2']).abs().argsort()[:1]]['Corrected'].values[0]
            sn = extract_sample_number(fname)
            if sn in conc_map:
                results.append((conc_map[sn], i1 / i2))

    df = pd.DataFrame(results, columns=['den_concentration', 'Fluorescence Ratio']).sort_values('den_concentration')
    fit_and_plot(df, 'Fluorescence Ratio', config, 'Fluorescence Ratio')

def run_single_wavelength_analysis(config):
    results = []
    concentration_file = config['concentration_file'] or os.path.join(config['folder'], 'concentrations.txt')
    conc_df = pd.read_csv(concentration_file, sep="\t")
    conc_map = conc_df.set_index("Sample_number")["den_concentration"]

    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(config['folder'], fname), header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])
            df.dropna(inplace=True)
            df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'], config['smoothing'], config['window'], config['spline_s'], config['poly'])
            baseline = np.interp(config['baseline'], df['Wavelength'], df['Smoothed']) if config['baseline'] else 0
            df['Corrected'] = df['Smoothed'] - baseline

            intensity = df.iloc[(df['Wavelength'] - config['target_wl']).abs().argsort()[:1]]['Corrected'].values[0]
            sn = extract_sample_number(fname)
            if sn in conc_map:
                results.append((conc_map[sn], intensity))

    df = pd.DataFrame(results, columns=['den_concentration', 'Intensity']).sort_values('den_concentration')
    fit_and_plot(df, 'Intensity', config, f'Intensity at {config["target_wl"]} nm (a.u.)')

def fit_and_plot(df, col, config, ylabel):
    x = df['den_concentration'].values
    y = df[col].values
    if config['fit_model'] == "None":
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label='Data', color='blue')
        plt.xlabel('Denaturant Concentration (M)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Denaturant Concentration')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config['folder'], f"{col}_nofit.png"))
        plt.show()
        return

    model = config['fit_model']
    guess = guess_initial_params(x, y, model)
    try:
        if model == "two_state":
            popt, pcov = curve_fit(G, x, y, p0=guess)
            y_fit = G(x, *popt)
        else:
            popt, pcov = curve_fit(G_three_state_weighted, x, y, p0=guess)
            y_fit = G_three_state_weighted(x, *popt)
        perr = np.sqrt(np.diag(pcov))
        print("Fitted parameters:")
        for i, (val, err) in enumerate(zip(popt, perr)):
            print(f"  Param {i + 1}: {val:.4f} ± {err:.4f}")
    except Exception as e:
        print("Fitting failed:", e)
        popt = None
        y_fit = None

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data', color='blue')
    if y_fit is not None:
        plt.plot(x, y_fit, label='Fit', color='red')
    plt.xlabel('Denaturant Concentration (M)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Denaturant Concentration')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['folder'], f"{col}_fit.png"))
    plt.show()

def main():
    config = {
        'folder': "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/22h/fluo/spectra_83_F4N_JS_26_04",  # <-- update this
        'concentration_file': None,
        'smoothing': "savitzky_golay",
        'window': 10,
        'spline_s': 0.5,
        'poly': 3,
        'baseline': None,
        'fit_model': "two_state",  # Options: "two_state", "three_state", "None"
        'wl1': 325,
        'wl2': 355,
        'target_wl': 325,
        'method': "all"  # Options: "all", "ew", "ratio", "single_wavelength"
    }

    if config['method'] in ["all", "ew"]:
        run_ew_analysis(config)
    if config['method'] in ["all", "ratio"]:
        run_ratio_analysis(config)
    if config['method'] in ["all", "single_wavelength"]:
        run_single_wavelength_analysis(config)

if __name__ == "__main__":
    main()
