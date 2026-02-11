import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

RT = 0.592  # RT constant kcal/mol


# --- MODEL DEFINITIONS ---
def G(x, a_n, a_u, m, d):
    """Two-state equilibrium unfolding model."""
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


def G_three_state_weighted(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """Three-state equilibrium unfolding model."""
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


# --- UTILITIES ---
def calculate_window_points(wavelength_array, nm_interval):
    """Converts a nanometer interval (e.g., 10nm) into number of data points."""
    step_size = wavelength_array.diff().median()
    if pd.isna(step_size) or step_size == 0:
        return 5

    window_points = int(np.ceil(nm_interval / step_size))
    # Ensure window is odd for Savitzky-Golay and at least 3
    if window_points % 2 == 0:
        window_points += 1
    return max(3, window_points)


def smooth_data(x, y, method, window_nm, spline_smoothing_factor, poly_order):
    window_size = calculate_window_points(x, window_nm)

    if method == "moving_average":
        return pd.Series(y).rolling(window=window_size, center=True).mean().to_numpy()
    elif method == "spline":
        return UnivariateSpline(x, y, s=spline_smoothing_factor)(x)
    elif method == "savitzky_golay":
        return savgol_filter(y, window_length=window_size, polyorder=poly_order)
    return y


def extract_sample_number(name):
    # Grabs the last number found in the filename to handle prefixes like "protein_1.csv"
    match = re.findall(r'(\d+)', name)
    return int(match[-1]) if match else None


def load_and_preprocess(fname, config):
    try:
        path = os.path.join(config['folder'], fname)
        df = pd.read_csv(path, header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])

        df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors='coerce')
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors='coerce')
        df.dropna(inplace=True)

        if df.empty: return None

        df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'],
                                     config['smoothing'], config['window'],
                                     config['spline_s'], config['poly'])

        # Drop NaNs created by smoothing BEFORE baseline calculation
        df.dropna(subset=['Smoothed'], inplace=True)

        if config['baseline']:
            base_idx = (df['Wavelength'] - config['baseline']).abs().idxmin()
            baseline_val = df.loc[base_idx, 'Smoothed']
            df['Corrected'] = df['Smoothed'] - baseline_val
        else:
            df['Corrected'] = df['Smoothed']

        return df
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None


# --- ANALYSIS MODULES ---

def run_csm_analysis(config, conc_map):
    """Calculates Center of Spectral Mass (Average Emission Wavelength)."""
    results = []
    print("Running Center of Spectral Mass (CSM) Analysis...")

    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            sn = extract_sample_number(fname)
            if sn not in conc_map: continue

            df = load_and_preprocess(fname, config)
            if df is None: continue

            # CSM Calculation: Sum(I * lambda) / Sum(I)
            weighted_sum = (df['Corrected'] * df['Wavelength']).sum()
            total_intensity = df['Corrected'].sum()

            if total_intensity == 0:
                csm = 0
            else:
                csm = weighted_sum / total_intensity

            results.append((conc_map[sn], csm))

    df_res = pd.DataFrame(results, columns=['den_concentration', 'CSM']).sort_values('den_concentration')
    fit_and_plot(df_res, 'CSM', config, 'Average Emission Wavelength (nm)')


def run_ratio_analysis(config, conc_map):
    results = []
    print("Running Ratio Analysis...")
    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            sn = extract_sample_number(fname)
            if sn not in conc_map: continue

            df = load_and_preprocess(fname, config)
            if df is None: continue

            try:
                i1 = df.iloc[(df['Wavelength'] - config['wl1']).abs().argsort()[:1]]['Corrected'].values[0]
                i2 = df.iloc[(df['Wavelength'] - config['wl2']).abs().argsort()[:1]]['Corrected'].values[0]
                results.append((conc_map[sn], i1 / i2))
            except IndexError:
                print(f"Wavelengths {config['wl1']} or {config['wl2']} not found in {fname}")

    df_res = pd.DataFrame(results, columns=['den_concentration', 'Fluorescence Ratio']).sort_values('den_concentration')
    fit_and_plot(df_res, 'Fluorescence Ratio', config, f'Ratio {config["wl1"]}/{config["wl2"]} nm')


def run_single_wavelength_analysis(config, conc_map):
    results = []
    print("Running Single Wavelength Analysis...")
    for fname in os.listdir(config['folder']):
        if fname.endswith(".csv"):
            sn = extract_sample_number(fname)
            if sn not in conc_map: continue

            df = load_and_preprocess(fname, config)
            if df is None: continue

            try:
                intensity = df.iloc[(df['Wavelength'] - config['target_wl']).abs().argsort()[:1]]['Corrected'].values[0]
                results.append((conc_map[sn], intensity))
            except IndexError:
                print(f"Wavelength {config['target_wl']} not found in {fname}")

    df_res = pd.DataFrame(results, columns=['den_concentration', 'Intensity']).sort_values('den_concentration')
    fit_and_plot(df_res, 'Intensity', config, f'Intensity at {config["target_wl"]} nm (a.u.)')


def fit_and_plot(df, col, config, ylabel):
    if df.empty:
        print(f"No data collected for {col}")
        return

    x = df['den_concentration'].values
    y = df[col].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data', color='blue', s=40, edgecolors='white')

    y_fit = None
    if config['fit_model'] != "None":
        try:
            guess = guess_initial_params(x, y, config['fit_model'])

            # Define parameter names based on model
            if config['fit_model'] == "two_state":
                # User requested specific naming for 2-state
                param_names = ["a_n", "a_u", "m", "d"]
                popt, pcov = curve_fit(G, x, y, p0=guess, maxfev=5000)
                y_fit = G(x, *popt)
            elif config['fit_model'] == "three_state":
                param_names = ["a_n", "a_i", "a_u", "m1", "d1", "m2", "d2"]
                popt, pcov = curve_fit(G_three_state_weighted, x, y, p0=guess, maxfev=5000)
                y_fit = G_three_state_weighted(x, *popt)

            sort_idx = np.argsort(x)
            plt.plot(x[sort_idx], y_fit[sort_idx], label='Fit', color='red', linewidth=2)

            perr = np.sqrt(np.diag(pcov))
            print(f"\n--- Fitted parameters for {col} ---")
            for i, (val, err) in enumerate(zip(popt, perr)):
                # Map index to name if available, else generic
                name = param_names[i] if i < len(param_names) else f"Param {i + 1}"
                print(f"  {name}: {val:.4f} ± {err:.4f}")

        except Exception as e:
            print(f"Fitting failed for {col}: {e}")

    plt.xlabel('Denaturant Concentration (M)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs [Denaturant]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(config['folder'], f"{col}_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    config = {
        'folder': "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/fluorimetry/TrmD",
        'concentration_file': "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/fluorimetry/TrmD/concentrations.txt",

        # Smoothing
        'smoothing': "savitzky_golay",
        'window': 10,  # 10 nm interval
        'spline_s': 0.5,
        'poly': 3,

        # Analysis
        'baseline': 400,  # Set to e.g., 400 if needed
        'fit_model': "two_state",
        'wl1': 325,  # For Ratio
        'wl2': 350,  # For Ratio
        'target_wl': 325,  # For Single Wavelength

        # Method options: "csm" (was ew), "ratio", "single_wavelength", "all"
        'method': "all"
    }

    # Load concentrations
    concentration_file = config['concentration_file'] or os.path.join(config['folder'], 'concentrations.txt')
    try:
        conc_df = pd.read_csv(concentration_file, sep="\t")
        conc_map = dict(zip(conc_df["Sample_number"], conc_df["den_concentration"]))
    except Exception as e:
        print(f"Failed to load concentrations: {e}")
        return

    # Run selected analyses
    if config['method'] in ["all", "csm"]:
        run_csm_analysis(config, conc_map)
    if config['method'] in ["all", "ratio"]:
        run_ratio_analysis(config, conc_map)
    if config['method'] in ["all", "single_wavelength"]:
        run_single_wavelength_analysis(config, conc_map)


if __name__ == "__main__":
    main()