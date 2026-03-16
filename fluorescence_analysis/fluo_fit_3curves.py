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
    match = re.findall(r'(\d+)', name)
    return int(match[-1]) if match else None


def load_and_preprocess(fname, folder, config):
    try:
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])

        df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors='coerce')
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors='coerce')
        df.dropna(inplace=True)

        if df.empty: return None

        df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'],
                                     config['smoothing'], config['window'],
                                     config['spline_s'], config['poly'])

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

def extract_series_data(folder, conc_map, config, series_name):
    """Extracts CSM, Ratio, and Single Wavelength data for a given folder in one pass."""
    csm_results = []
    ratio_results = []
    single_wl_results = []

    print(f"Extracting data for {series_name} from {folder}...")

    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        sn = extract_sample_number(fname)
        if sn not in conc_map:
            continue

        df = load_and_preprocess(fname, folder, config)
        if df is None:
            continue

        # 1. CSM Calculation (Filtered by range)
        csm_min = config.get('csm_min', 320)
        csm_max = config.get('csm_max', 400)
        csm_df = df[(df['Wavelength'] >= csm_min) & (df['Wavelength'] <= csm_max)]

        weighted_sum = (csm_df['Corrected'] * csm_df['Wavelength']).sum()
        total_intensity = csm_df['Corrected'].sum()
        csm = weighted_sum / total_intensity if total_intensity != 0 else 0
        csm_results.append((conc_map[sn], csm))

        # 2. Ratio Calculation
        try:
            i1 = df.iloc[(df['Wavelength'] - config['wl1']).abs().argsort()[:1]]['Corrected'].values[0]
            i2 = df.iloc[(df['Wavelength'] - config['wl2']).abs().argsort()[:1]]['Corrected'].values[0]
            ratio_results.append((conc_map[sn], i1 / i2))
        except IndexError:
            pass

        # 3. Single Wavelength Calculation
        try:
            intensity = df.iloc[(df['Wavelength'] - config['target_wl']).abs().argsort()[:1]]['Corrected'].values[0]
            single_wl_results.append((conc_map[sn], intensity))
        except IndexError:
            pass

    # Convert to sorted DataFrames
    df_csm = pd.DataFrame(csm_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')
    df_ratio = pd.DataFrame(ratio_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')
    df_single = pd.DataFrame(single_wl_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')

    return {'CSM': df_csm, 'Ratio': df_ratio, 'SingleWL': df_single}


def compare_and_plot(df1, df2, df3, title, ylabel, save_name, base_path, config):
    plt.figure(figsize=(8, 6))

    def plot_series(df, name, color, marker, linestyle):
        if df is None or df.empty:
            print(f"No data available for {name} ({title}).")
            return

        x = df['den_concentration'].values
        y = df['Value'].values

        plt.scatter(x, y, label=f'{name} Data', color=color, marker=marker, s=40, edgecolors='white')

        if config['fit_model'] != "None":
            try:
                guess = guess_initial_params(x, y, config['fit_model'])

                if config['fit_model'] == "two_state":
                    param_names = ["a_n", "a_u", "m", "d"]
                    popt, pcov = curve_fit(G, x, y, p0=guess, maxfev=5000)
                    y_fit = G(x, *popt)
                elif config['fit_model'] == "three_state":
                    param_names = ["a_n", "a_i", "a_u", "m1", "d1", "m2", "d2"]
                    popt, pcov = curve_fit(G_three_state_weighted, x, y, p0=guess, maxfev=5000)
                    y_fit = G_three_state_weighted(x, *popt)

                sort_idx = np.argsort(x)
                plt.plot(x[sort_idx], y_fit[sort_idx], label=f'{name} Fit', color=color, linestyle=linestyle,
                         linewidth=2)

                perr = np.sqrt(np.diag(pcov))
                print(f"\n--- Fitted parameters for {title} [{name}] ---")
                for i, (val, err) in enumerate(zip(popt, perr)):
                    p_name = param_names[i] if i < len(param_names) else f"Param {i + 1}"
                    print(f"  {p_name}: {val:.4f} ± {err:.4f}")

            except Exception as e:
                print(f"Fitting failed for {name} ({title}): {e}")

    # Plot all three series
    plot_series(df1, 'series1', 'blue', 'o', '-')
    plot_series(df2, 'series2', 'red', 's', '--')
    plot_series(df3, 'series3', 'green', '^', ':')

    plt.xlabel('Denaturant Concentration (M)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(base_path, f"{save_name}_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    print("-" * 40)
    plt.show()


def load_concentrations(filepath):
    try:
        conc_df = pd.read_csv(filepath, sep="\t")
        return dict(zip(conc_df["Sample_number"], conc_df["den_concentration"]))
    except Exception as e:
        print(f"Failed to load concentrations from {filepath}: {e}")
        return {}


def main():
    # Base configuration parameters
    config = {
        # Smoothing
        'smoothing': "savitzky_golay",
        'window': 25,  # 10 nm interval
        'spline_s': 0.5,
        'poly': 3,

        # Analysis
        'baseline': None,  # Set to e.g., 400 if needed
        'fit_model': "two_state",

        # New CSM Parameters
        'csm_min': 320,  # CSM calculation lower bound (nm)
        'csm_max': 400,  # CSM calculation upper bound (nm)

        'wl1': 330,  # For Ratio
        'wl2': 350,  # For Ratio
        'target_wl': 330,  # For Single Wavelength

        # Method options: "csm", "ratio", "single_wavelength", "all"
        'method': "all"
    }

    # Define paths
    base_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/TrmD"

    path_series1 = os.path.join(base_path, "seria2_1")
    conc_series1 = os.path.join(path_series1, "concentrations.txt")

    path_series2 = os.path.join(base_path, "seria2_2")
    conc_series2 = os.path.join(path_series2, "concentrations.txt")

    # Added 3rd series path
    path_series3 = os.path.join(base_path, "seria2_3")
    conc_series3 = os.path.join(path_series3, "concentrations.txt")

    # Load Concentration Maps
    conc_map1 = load_concentrations(conc_series1)
    conc_map2 = load_concentrations(conc_series2)
    conc_map3 = load_concentrations(conc_series3)

    if not conc_map1 or not conc_map2 or not conc_map3:
        print("Missing concentration files for one or more series. Please check paths. Exiting.")
        return

    # Extract Data
    data1 = extract_series_data(path_series1, conc_map1, config, "seria2_1")
    data2 = extract_series_data(path_series2, conc_map2, config, "seria2_2")
    data3 = extract_series_data(path_series3, conc_map3, config, "seria2_3")

    # Generate Comparison Plots based on chosen method
    if config['method'] in ["all", "csm"]:
        compare_and_plot(data1['CSM'], data2['CSM'], data3['CSM'],
                         title='Center of Spectral Mass (CSM) Comparison',
                         ylabel='Average Emission Wavelength (nm)',
                         save_name='CSM', base_path=base_path, config=config)

    if config['method'] in ["all", "ratio"]:
        compare_and_plot(data1['Ratio'], data2['Ratio'], data3['Ratio'],
                         title=f'Fluorescence Ratio ({config["wl1"]}/{config["wl2"]} nm) Comparison',
                         ylabel=f'Ratio {config["wl1"]}/{config["wl2"]}',
                         save_name='Ratio', base_path=base_path, config=config)

    if config['method'] in ["all", "single_wavelength"]:
        compare_and_plot(data1['SingleWL'], data2['SingleWL'], data3['SingleWL'],
                         title=f'Single Wavelength Intensity ({config["target_wl"]} nm) Comparison',
                         ylabel='Intensity (a.u.)',
                         save_name='Single_Wavelength', base_path=base_path, config=config)


if __name__ == "__main__":
    main()