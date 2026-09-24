import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

RT = 0.592  # kcal/mol at 298.15 K

# --- PUBLICATION PALETTE & STYLES ---
# Consistent styling across all figures in your paper
PROTEIN_STYLES = {
    'TrmD': {
        'color': '#1F77B4',       # Deep Royal Blue
        'marker': 'o',
        'linestyle': '-',
        'label': 'TrmD'
    },
    'Tm1570': {
        'color': '#D95F02',       # Warm Vermilion / Rust
        'marker': 's',
        'linestyle': '--',
        'label': 'Tm1570'
    },
    'TrmD-Tm1570': {
        'color': '#7570B3',       # Muted Purple / Orchid (for your 3rd fusion protein)
        'marker': '^',
        'linestyle': '-.',
        'label': 'TrmD-Tm1570'
    }
}


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
        a_n = np.mean(y_sorted[:max(1, len(x) // 3)])
        a_u = np.mean(y_sorted[-max(1, len(x) // 3):])
        dy_dx = np.gradient(y_sorted, x_sorted)
        d = x_sorted[np.argmax(np.abs(dy_dx))]
        m = 5 / max(np.ptp(x), 1e-6)
        return [a_n, a_u, m, d]

    elif model == "three_state":
        n = len(x_sorted)
        third = max(1, n // 3)
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
    """Converts a nanometer interval into number of data points."""
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


def load_concentrations(filepath, min_conc=None, max_conc=None):
    """Loads concentrations and filters by optional min/max limits."""
    try:
        try:
            conc_df = pd.read_csv(filepath, sep=r'\s+')
        except Exception:
            conc_df = pd.read_csv(filepath, sep="\t")

        conc_df["Sample_number"] = pd.to_numeric(conc_df["Sample_number"], errors='coerce')
        conc_df["den_concentration"] = pd.to_numeric(conc_df["den_concentration"], errors='coerce')
        conc_df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
        conc_df["Sample_number"] = conc_df["Sample_number"].astype(int)

        if min_conc is not None:
            conc_df = conc_df[conc_df["den_concentration"] >= min_conc]
        if max_conc is not None:
            conc_df = conc_df[conc_df["den_concentration"] <= max_conc]

        return dict(zip(conc_df["Sample_number"], conc_df["den_concentration"]))
    except Exception as e:
        print(f"Failed to load concentrations from {filepath}: {e}")
        return {}


def load_and_preprocess(fname, folder, config):
    try:
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])
        df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors='coerce')
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return None

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

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        sn = extract_sample_number(fname)
        if sn is None or sn not in conc_map:
            continue

        df = load_and_preprocess(fname, folder, config)
        if df is None:
            continue

        # 1. CSM Calculation
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

    df_csm = pd.DataFrame(csm_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')
    df_ratio = pd.DataFrame(ratio_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')
    df_single = pd.DataFrame(single_wl_results, columns=['den_concentration', 'Value']).sort_values('den_concentration')

    return {'CSM': df_csm, 'Ratio': df_ratio, 'SingleWL': df_single}


def compare_and_plot(series_list, title, ylabel, save_name, base_path, config):
    """
    Plots a publication-quality figure panel.
    series_list: list of tuples -> [('Tm1570', df1), ('TrmD', df2)]
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)

    for name, df in series_list:
        if df is None or df.empty:
            print(f"No data available for {name} ({title}).")
            continue

        style = PROTEIN_STYLES.get(name, {
            'color': '#333333', 'marker': 'o', 'linestyle': '-', 'label': name
        })

        x = df['den_concentration'].values
        y = df['Value'].values

        # Plot raw experimental data points
        ax.scatter(x, y, label=f"{style['label']}",
                   color=style['color'], marker=style['marker'],
                   s=55, alpha=0.9, edgecolors=None, linewidths=0, zorder=3)

        # Fit model and plot high-resolution smooth sigmoidal curve
        if config.get('fit_model', 'None') != "None":
            try:
                guess = guess_initial_params(x, y, config['fit_model'])
                x_fit = np.linspace(np.min(x), np.max(x), 400)

                if config['fit_model'] == "two_state":
                    param_names = ["a_n", "a_u", "m", "C_mid"]
                    popt, pcov = curve_fit(G, x, y, p0=guess, maxfev=10000)
                    y_fit = G(x_fit, *popt)
                elif config['fit_model'] == "three_state":
                    param_names = ["a_n", "a_i", "a_u", "m1", "C_mid1", "m2", "C_mid2"]
                    popt, pcov = curve_fit(G_three_state_weighted, x, y, p0=guess, maxfev=10000)
                    y_fit = G_three_state_weighted(x_fit, *popt)

                ax.plot(x_fit, y_fit, color=style['color'],
                        linestyle=style['linestyle'], linewidth=2.2, zorder=2)

                perr = np.sqrt(np.diag(pcov))
                print(f"\n--- Fit parameters for {title} [{name}] ---")
                for i, (val, err) in enumerate(zip(popt, perr)):
                    p_name = param_names[i] if i < len(param_names) else f"Param {i + 1}"
                    print(f"  {p_name}: {val:.4f} ± {err:.4f}")

            except Exception as e:
                print(f"Fitting failed for {name} ({title}): {e}")

    # Publication-Grade Panel Typography & Axis Formatting
    ax.set_xlabel('[GuCl] (M)', fontsize=16, labelpad=8, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8, fontweight='medium')

    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2, direction='in')
    ax.tick_params(axis='both', which='minor', length=3.5, width=1.0, direction='in')

    # Despine: standard clean top/right spine removal for biochemistry panels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Optional subtle horizontal grid or leave blank for crisp aesthetic
    # ax.grid(True, linestyle=':', alpha=0.4, color='gray')

    # Legend formatting
    legend = ax.legend(frameon=False, framealpha=0.9, edgecolor='none',
                       fontsize=13, loc='best')

    plt.tight_layout()

    # Save both 300 DPI raster PNG and vector PDF for multi-panel assembly
    save_base = os.path.join(base_path, save_name)
    plt.savefig(f"{save_base}.png", dpi=300)
    plt.savefig(f"{save_base}.pdf", format="pdf")
    print(f"\nSaved panel figures:\n  -> {save_base}.png\n  -> {save_base}.pdf")
    print("-" * 50)
    plt.show()


def main():
    config = {
        # Smoothing
        'smoothing': "savitzky_golay",
        'window': 25,
        'spline_s': 0.5,
        'poly': 3,

        # Baseline & Fitting
        'baseline': 400,
        'fit_model': "two_state",  # "two_state", "three_state", or "None"

        # Concentration limits (Set to None for full range)
        'min_conc': None,
        'max_conc': None,

        # CSM Bounds
        'csm_min': 320,
        'csm_max': 400,

        # Wavelengths
        'wl1': 330,
        'wl2': 350,
        'target_wl': 335,

        # Method options: "csm", "ratio", "single_wavelength", "all"
        'method': "single_wavelength"
    }

    base_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/"

    path_tm1570 = os.path.join(base_path, "Tm1570/3/2uM/28.07.26/csv")
    conc_tm1570 = os.path.join(path_tm1570, "concentrations.txt")

    path_trmd = os.path.join(base_path, "TrmD/3/2uM/28.07.26/csv")
    conc_trmd = os.path.join(path_trmd, "concentrations.txt")

    # Load Concentration Maps
    conc_map_tm1570 = load_concentrations(conc_tm1570, config['min_conc'], config['max_conc'])
    conc_map_trmd = load_concentrations(conc_trmd, config['min_conc'], config['max_conc'])

    if not conc_map_tm1570 or not conc_map_trmd:
        print("Missing concentration files or no valid data loaded. Check paths. Exiting.")
        return

    # Extract Datasets
    data_tm1570 = extract_series_data(path_tm1570, conc_map_tm1570, config, "Tm1570")
    data_trmd = extract_series_data(path_trmd, conc_map_trmd, config, "TrmD")

    # When you add the fusion later, simply extract it here:
    # data_fusion = extract_series_data(path_fusion, conc_map_fusion, config, "TrmD-Tm1570")

    # Generate Comparison Plots
    if config['method'] in ["all", "csm"]:
        compare_and_plot(
            [('Tm1570', data_tm1570['CSM']), ('TrmD', data_trmd['CSM'])],
            title='Center of Spectral Mass (CSM)',
            ylabel='Emission CSM (nm)',
            save_name='CSM_TrmD_vs_Tm1570',
            base_path=base_path, config=config
        )

    if config['method'] in ["all", "ratio"]:
        compare_and_plot(
            [('Tm1570', data_tm1570['Ratio']), ('TrmD', data_trmd['Ratio'])],
            title=f'Fluorescence Ratio ({config["wl1"]}/{config["wl2"]} nm)',
            ylabel=f'$I_{{{config["wl1"]}}} / I_{{{config["wl2"]}}}$',
            save_name='Ratio_TrmD_vs_Tm1570',
            base_path=base_path, config=config
        )

    if config['method'] in ["all", "single_wavelength"]:
        compare_and_plot(
            [('Tm1570', data_tm1570['SingleWL']), ('TrmD', data_trmd['SingleWL'])],
            title=f'Intensity at {config["target_wl"]} nm',
            ylabel=f'Intensity at {config["target_wl"]} nm (a.u.)',
            save_name='Single_WL_TrmD_vs_Tm1570',
            base_path=base_path, config=config
        )


if __name__ == "__main__":
    main()