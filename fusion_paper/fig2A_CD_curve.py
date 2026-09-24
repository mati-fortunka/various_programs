import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # kcal/mol at 298.15 K

# --- PUBLICATION PALETTE & STYLES (Identical to Fluorimetry Panel) ---
PROTEIN_STYLES = {
    'TrmD': {
        'color': '#1F77B4',  # Deep Royal Blue
        'marker': 'o',
        'linestyle': '-',
        'label': 'TrmD'
    },
    'Tm1570': {
        'color': '#D95F02',  # Warm Vermilion / Rust
        'marker': 's',
        'linestyle': '--',
        'label': 'Tm1570'
    },
    'TrmD-Tm1570': {
        'color': '#7570B3',  # Muted Purple / Orchid
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


# --- HELPER FUNCTIONS ---
def find_block(lines, property_name):
    """Locates block boundaries in JASCO/standard CD spectrometer outputs."""
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith(property_name):
            start = i + 2
            end = start
            while end < len(lines) and lines[end].strip():
                if lines[end].strip().startswith("Wavelength"):
                    break
                end += 1
            return start, end
    return None, None


def extract_sample_number(name):
    """Extracts numerical sample ID from file name."""
    matches = re.findall(r'(\d+)', name)
    return int(matches[-1]) if matches else None


# --- MAIN CD EXTRACTION & FITTING ---
def process_cd_data(folder_path, wavelength, concentration_file, series_name="Series",
                    smoothing_method="savitzky_golay", window_size=15, spline_smoothing_factor=0.5,
                    poly_order=3, baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
                    model="2-state", min_conc=None, max_conc=None, custom_p0=None):
    with open(concentration_file, 'r') as f:
        conc_text = f.read().replace(',', '.')

    concentration_data = pd.read_csv(StringIO(conc_text), sep=r"\s+", engine='python')
    concentration_data['den_concentration'] = pd.to_numeric(concentration_data['den_concentration'], errors='coerce')
    concentration_data.dropna(subset=['Sample_number', 'den_concentration'], inplace=True)

    # Optional concentration limits filtering
    if min_conc is not None:
        concentration_data = concentration_data[concentration_data['den_concentration'] >= min_conc]
    if max_conc is not None:
        concentration_data = concentration_data[concentration_data['den_concentration'] <= max_conc]

    concentration_mapping = dict(zip(concentration_data['Sample_number'].astype(int),
                                     concentration_data['den_concentration']))

    ellipticity_vs_concentration = []

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".csv"):
            continue

        sample_number = extract_sample_number(file_name)
        if sample_number is None or sample_number not in concentration_mapping:
            continue

        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            cd_start, cd_end = find_block(lines, "CircularDichroism")
            if cd_start is None:
                continue

            cd_lines = [ln.strip() for ln in lines[cd_start:cd_end] if ln.strip()]
            cd_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in cd_lines]
            cd_data = np.array(cd_pairs)
            if cd_data.shape[1] < 2:
                continue

            wavelengths_cd = cd_data[:, 0]
            ellipticity = cd_data[:, 1]

            # HV signal reading & cutoff
            hv_start, hv_end = find_block(lines, "HV")
            if hv_start is not None:
                hv_lines = [ln.strip() for ln in lines[hv_start:hv_end] if ln.strip()]
                hv_data = np.array([list(map(float, ln.replace(',', ' ').split())) for ln in hv_lines])
                wavelengths_hv = hv_data[:, 0]
                hv_values = hv_data[:, 1]

                if not np.allclose(wavelengths_cd, wavelengths_hv):
                    common, idx_cd, idx_hv = np.intersect1d(wavelengths_cd, wavelengths_hv, return_indices=True)
                    wavelengths_cd = wavelengths_cd[idx_cd]
                    ellipticity = ellipticity[idx_cd]
                    hv_values = hv_values[idx_hv]

                if hv_cutoff is not None and hv_mode == 'per_point':
                    mask = np.where(np.isnan(hv_values), True, hv_values <= hv_cutoff)
                    wavelengths_cd = wavelengths_cd[mask]
                    ellipticity = ellipticity[mask]

            if len(wavelengths_cd) == 0:
                continue

            # Smoothing
            smoothed = ellipticity.copy()
            if smoothing_method == "savitzky_golay" and len(ellipticity) >= window_size:
                smoothed = savgol_filter(ellipticity, window_length=window_size, polyorder=poly_order)
            elif smoothing_method == "spline" and len(wavelengths_cd) >= 4:
                smoothed = UnivariateSpline(wavelengths_cd, ellipticity, s=spline_smoothing_factor)(wavelengths_cd)

            if wavelengths_cd[0] > wavelengths_cd[-1]:
                wavelengths_cd = wavelengths_cd[::-1]
                smoothed = smoothed[::-1]

            baseline_value = np.interp(baseline_wavelength, wavelengths_cd, smoothed) if baseline_wavelength else 0.0
            target_ellipticity = np.interp(wavelength, wavelengths_cd, smoothed)
            corrected_ellipticity = target_ellipticity - baseline_value

            den_conc = concentration_mapping[sample_number]
            ellipticity_vs_concentration.append((float(den_conc), float(corrected_ellipticity)))

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if not ellipticity_vs_concentration:
        print(f"Warning: No valid data points extracted for {series_name}.")
        return None, None

    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Ellipticity'].values

    # Model fitting
    popt, pcov = None, None
    try:
        if model in ["3-state", "three_state"]:
            fit_func = G_three_state_weighted
            param_names = ["a_n", "a_i", "a_u", "m1", "C_mid1", "m2", "C_mid2"]
            if custom_p0 is not None:
                initial_guess = custom_p0
            else:
                a_n_init = np.mean(y_data[:max(1, len(y_data) // 4)])
                a_u_init = np.mean(y_data[-max(1, len(y_data) // 4):])
                a_i_init = (a_n_init + a_u_init) / 2.0
                initial_guess = [a_n_init, a_i_init, a_u_init, 2.0, 2.0, 2.0, 3.5]

            param_bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, 0, 0],
                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        else:
            fit_func = G
            param_names = ["a_n", "a_u", "m", "C_mid"]
            if custom_p0 is not None:
                initial_guess = custom_p0
            else:
                a_n_init = np.mean(y_data[:max(1, len(y_data) // 4)])
                a_u_init = np.mean(y_data[-max(1, len(y_data) // 4):])
                dy_dx = np.gradient(y_data, x_data)
                d_init = x_data[np.argmax(np.abs(dy_dx))]
                initial_guess = [a_n_init, a_u_init, 2.0, d_init]

            param_bounds = (-np.inf, np.inf)

        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guess, bounds=param_bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        print(f"\n--- Fitted {model} parameters for {series_name} ---")
        for i, (val, err) in enumerate(zip(popt, perr)):
            print(f"  {param_names[i]}: {val:.4f} ± {err:.4f}")

    except Exception as e:
        print(f"Curve fitting failed for {series_name} ({model}): {e}")

    return plot_data, popt


# --- PUBLICATION PLOT FUNCTION ---
def plot_cd_comparison(series_list, wavelength, save_name, base_path):
    """
    Renders the CD equilibrium unfolding publication panel.
    series_list: list of dicts with keys: 'name', 'data', 'popt', 'model'
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)

    for item in series_list:
        name = item['name']
        df = item['data']
        popt = item['popt']
        model = item['model']

        if df is None or df.empty:
            continue

        style = PROTEIN_STYLES.get(name, {
            'color': '#333333', 'marker': 'o', 'linestyle': '-', 'label': name
        })

        x = df['den_concentration'].values
        y = df['Ellipticity'].values

        # Scatter points: edges completely deleted
        ax.scatter(x, y, label=style['label'],
                   color=style['color'], marker=style['marker'],
                   s=55, alpha=0.9, edgecolors='none', linewidths=0, zorder=3)

        # Plot fitted curve
        if popt is not None:
            x_fit = np.linspace(np.min(x), np.max(x), 400)
            if model in ["3-state", "three_state"]:
                y_fit = G_three_state_weighted(x_fit, *popt)
            else:
                y_fit = G(x_fit, *popt)

            ax.plot(x_fit, y_fit, color=style['color'],
                    linestyle=style['linestyle'], linewidth=2.2, zorder=2)

    # Typography & Axis Styling
    ax.set_xlabel('[GuCl] (M)', fontsize=16, labelpad=8, fontweight='medium')
    ax.set_ylabel(f'Ellipticity at {wavelength} nm (mdeg)', fontsize=16, labelpad=8, fontweight='medium')

    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2, direction='in')
    ax.tick_params(axis='both', which='minor', length=3.5, width=1.0, direction='in')

    # Despine: clean top and right spine removal
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Legend without box outline
    ax.legend(frameon=False, fontsize=13, loc='best')

    plt.tight_layout()

    # Save both vector PDF and 300 DPI PNG
    save_base = os.path.join(base_path, save_name)
    plt.savefig(f"{save_base}.png", dpi=300)
    plt.savefig(f"{save_base}.pdf", format="pdf")
    print(f"\nSaved CD figure panel:\n  -> {save_base}.png\n  -> {save_base}.pdf")
    print("-" * 50)
    plt.show()


# --- EXECUTION ---
if __name__ == "__main__":
    base_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/"

    path_trmd = os.path.join(base_path, "trmd/3/2uM/28_07_26")
    conc_trmd = os.path.join(path_trmd, "concentrations.txt")

    path_tm1570 = os.path.join(base_path, "tm1570/3/2uM/28_07_26")
    conc_tm1570 = os.path.join(path_tm1570, "concentrations.txt")

    path_fusion = os.path.join(base_path, "fuzja/3/2uM/28_07_26")
    conc_fusion = os.path.join(path_fusion, "concentrations.txt")

    wavelength_to_check = 217

    # 1. Process TrmD (2-state model)
    print("Processing TrmD...")
    data_trmd, popt_trmd = process_cd_data(
        path_trmd, wavelength_to_check, conc_trmd, series_name="TrmD",
        smoothing_method="savitzky_golay", window_size=15, poly_order=3,
        baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
        model="2-state"
    )

    # 2. Process Tm1570 (2-state model)
    print("Processing Tm1570...")
    data_tm1570, popt_tm1570 = process_cd_data(
        path_tm1570, wavelength_to_check, conc_tm1570, series_name="Tm1570",
        smoothing_method="savitzky_golay", window_size=15, poly_order=3,
        baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
        model="2-state"
    )

    # 3. Process Fusion (3-state model)
    print("Processing TrmD-Tm1570...")
    data_fusion, popt_fusion = process_cd_data(
        path_fusion, wavelength_to_check, conc_fusion, series_name="TrmD-Tm1570",
        smoothing_method="savitzky_golay", window_size=15, poly_order=3,
        baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
        model="3-state"
    )

    # Assemble series and generate the combined comparison figure
    series_payload = [
        {'name': 'TrmD', 'data': data_trmd, 'popt': popt_trmd, 'model': '2-state'},
        {'name': 'Tm1570', 'data': data_tm1570, 'popt': popt_tm1570, 'model': '2-state'},
        {'name': 'TrmD-Tm1570', 'data': data_fusion, 'popt': popt_fusion, 'model': '3-state'},
    ]

    plot_cd_comparison(
        series_payload,
        wavelength=wavelength_to_check,
        save_name=f"CD_{wavelength_to_check}nm_TrmD_Tm1570_Fusion",
        base_path=base_path
    )