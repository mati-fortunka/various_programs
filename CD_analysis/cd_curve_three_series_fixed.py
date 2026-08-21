import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # RT constant in kcal/mol (at ~298 K)


# 2-state model function
def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


# 3-state weighted model function
def G_three_state_weighted(x, a_n, a_i, a_u, m1, d1, m2, d2):
    sigmoid1 = 1 / (1 + np.exp(-(m1 * (x - d1)) / RT))
    sigmoid2 = 1 / (1 + np.exp(-(m2 * (x - d2)) / RT))
    return a_n * (1 - sigmoid1) + a_i * (sigmoid1 - sigmoid2) + a_u * sigmoid2


def fit_fixed_3state(x_data, y_data, fixed_params=None, initial_guess=None):
    """
    Fits G_three_state_weighted allowing specific parameters to be fixed to constant values.
    """
    param_names = ['a_n', 'a_i', 'a_u', 'm1', 'd1', 'm2', 'd2']
    if fixed_params is None:
        fixed_params = {}

    free_params = [p for p in param_names if p not in fixed_params]

    default_p0 = {
        'a_n': y_data[0] if len(y_data) > 0 else -10.0,
        'a_i': (y_data[0] + y_data[-1]) / 2.0 if len(y_data) > 0 else -5.0,
        'a_u': y_data[-1] if len(y_data) > 0 else -1.0,
        'm1': 1.0, 'd1': 2.0,
        'm2': 1.5, 'd2': 3.0
    }

    if initial_guess is not None:
        for name, val in initial_guess.items():
            default_p0[name] = val

    p0_free = [default_p0[p] for p in free_params]

    def wrapped_func(x, *free_vals):
        full_kwargs = fixed_params.copy()
        full_kwargs.update(dict(zip(free_params, free_vals)))
        return G_three_state_weighted(x, **full_kwargs)

    popt_free, pcov_free = curve_fit(wrapped_func, x_data, y_data, p0=p0_free, maxfev=10000)

    full_popt = []
    perr_free = np.sqrt(np.diag(pcov_free))
    full_perr = []
    free_idx = 0

    for name in param_names:
        if name in fixed_params:
            full_popt.append(fixed_params[name])
            full_perr.append(0.0)
        else:
            full_popt.append(popt_free[free_idx])
            full_perr.append(perr_free[free_idx])
            free_idx += 1

    return np.array(full_popt), np.array(full_perr)


# Helper: find data block
def find_block(lines, property_name):
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


# Main processing function
def process_cd_data(folder_path, wavelength, concentration_file, series_name="Series",
                    smoothing_method=None, window_size=15, spline_smoothing_factor=0.5,
                    poly_order=3, baseline_wavelength=None, hv_cutoff=700, hv_mode='per_point',
                    model="3-state", fixed_params=None, custom_p0=None):

    with open(concentration_file, 'r') as f:
        conc_text = f.read().replace(',', '.')

    concentration_data = pd.read_csv(StringIO(conc_text), sep=r"\s+", engine='python')
    concentration_data['den_concentration'] = pd.to_numeric(concentration_data['den_concentration'], errors='coerce')
    concentration_mapping = concentration_data.set_index('Sample_number')['den_concentration']

    ellipticity_vs_concentration = []
    all_spectra = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            cd_start, cd_end = find_block(lines, "CircularDichroism")
            if cd_start is None:
                raise ValueError(f"CircularDichroism block not found in {file_name}")
            cd_lines = [ln.strip() for ln in lines[cd_start:cd_end] if ln.strip()]
            cd_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in cd_lines]
            cd_data = np.array(cd_pairs)
            if cd_data.shape[1] < 2:
                raise ValueError(f"Unexpected CD data shape in {file_name}: {cd_data.shape}")

            wavelengths_cd = cd_data[:, 0]
            ellipticity = cd_data[:, 1]

            hv_start, hv_end = find_block(lines, "HV")
            if hv_start is None:
                hv_values = np.full_like(wavelengths_cd, fill_value=np.nan, dtype=float)
            else:
                hv_lines = [ln.strip() for ln in lines[hv_start:hv_end] if ln.strip()]
                hv_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in hv_lines]
                hv_data = np.array(hv_pairs)
                if hv_data.shape[1] < 2:
                    raise ValueError(f"Unexpected HV data shape in {file_name}: {hv_data.shape}")
                wavelengths_hv = hv_data[:, 0]
                hv_values = hv_data[:, 1]

                if wavelengths_cd.shape != wavelengths_hv.shape or not np.allclose(wavelengths_cd, wavelengths_hv):
                    common, idx_cd, idx_hv = np.intersect1d(wavelengths_cd, wavelengths_hv, return_indices=True)
                    if common.size == 0:
                        raise ValueError(f"No matching wavelengths between CD and HV in {file_name}")
                    wavelengths_cd = wavelengths_cd[idx_cd]
                    ellipticity = ellipticity[idx_cd]
                    hv_values = hv_values[idx_hv]

            if hv_cutoff is not None and hv_mode:
                if hv_mode == 'per_point':
                    mask = np.where(np.isnan(hv_values), True, hv_values <= hv_cutoff)
                    wavelengths = wavelengths_cd[mask]
                    ellipticity = ellipticity[mask]
                    hv_values = hv_values[mask]
                elif hv_mode == 'per_spectrum':
                    if np.any(~np.isnan(hv_values) & (hv_values > hv_cutoff)):
                        print(f"Skipping {file_name} because HV exceeded cutoff.")
                        continue
                    else:
                        wavelengths = wavelengths_cd
                else:
                    raise ValueError("hv_mode must be 'per_point', 'per_spectrum', or None/False")
            else:
                wavelengths = wavelengths_cd

            if wavelengths.size == 0 or ellipticity.size == 0:
                continue

            smoothed = ellipticity.copy()
            if smoothing_method == "moving_average":
                if len(ellipticity) >= window_size:
                    smoothed = pd.Series(ellipticity).rolling(window=window_size, center=True).mean().to_numpy()
                    nan_idx = np.isnan(smoothed)
                    smoothed[nan_idx] = ellipticity[nan_idx]
            elif smoothing_method == "spline":
                if len(wavelengths) >= 4:
                    spline = UnivariateSpline(wavelengths, ellipticity, s=spline_smoothing_factor)
                    smoothed = spline(wavelengths)
            elif smoothing_method == "savitzky_golay":
                if len(ellipticity) >= window_size:
                    smoothed = savgol_filter(ellipticity, window_length=window_size, polyorder=poly_order)

            if wavelengths[0] > wavelengths[-1]:
                wavelengths = wavelengths[::-1]
                smoothed = smoothed[::-1]

            if baseline_wavelength is not None:
                baseline_value = np.interp(baseline_wavelength, wavelengths, smoothed)
            else:
                baseline_value = 0.0

            target_ellipticity = np.interp(wavelength, wavelengths, smoothed)
            corrected_ellipticity = target_ellipticity - baseline_value

            file_pattern = re.compile(r"(\d{5})\.csv$")
            match = file_pattern.search(file_name)
            if match:
                sample_number = int(match.group(1))
                den_conc = concentration_mapping.get(sample_number, None)
                if den_conc is not None and not np.isnan(den_conc):
                    ellipticity_vs_concentration.append((float(den_conc), float(corrected_ellipticity)))
                    all_spectra.append((wavelengths, smoothed, float(den_conc)))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not ellipticity_vs_concentration:
        raise RuntimeError(f"No data collected for {series_name}.")

    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values.astype(float)
    y_data = plot_data['Ellipticity'].values.astype(float)

    try:
        if model == "3-state":
            popt, perr = fit_fixed_3state(x_data, y_data, fixed_params=fixed_params, initial_guess=custom_p0)

            fit_results_text = (
                f"--- Fitted 3-state parameters for {series_name} ---\n"
                f"a_n = {popt[0]:.2f} ± {perr[0]:.2f}\n"
                f"a_i = {popt[1]:.2f} ± {perr[1]:.2f}\n"
                f"a_u = {popt[2]:.2f} ± {perr[2]:.2f}\n"
                f"m1  = {popt[3]:.2f} ± {perr[3]:.2f}\n"
                f"d1  = {popt[4]:.2f} ± {perr[4]:.2f} M\n"
                f"m2  = {popt[5]:.2f} ± {perr[5]:.2f}\n"
                f"d2  = {popt[6]:.2f} ± {perr[6]:.2f} M\n"
                f"-----------------------------------------"
            )
        else:
            p0 = [y_data[0], y_data[-1], 1.2, 2.0]
            popt, pcov = curve_fit(G, x_data, y_data, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            fit_results_text = f"--- Fitted 2-state parameters for {series_name} ---\n..."

        print("\n" + fit_results_text)
        fit_file_path = os.path.join(folder_path, "fit_fixed_3state.txt")
        with open(fit_file_path, "w") as fit_file:
            fit_file.write(fit_results_text)

    except Exception as e:
        print(f"Curve fitting failed for {series_name}: {e}")
        popt = None

    return plot_data, popt


# Example usage
if __name__ == "__main__":

    base_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/3/2uM/"

    path_series1 = os.path.join(base_path, "28_07_26")
    conc_series1 = os.path.join(path_series1, "concentrations.txt")

    path_series2 = os.path.join(base_path, "29_07_26")
    conc_series2 = os.path.join(path_series2, "concentrations.txt")

    path_series3 = os.path.join(base_path, "31_07_26")
    conc_series3 = os.path.join(path_series3, "concentrations.txt")

    wavelength_to_check = 217

    labels = {
        "s1": '12h',
        "s2": '36h',
        "s3": '84h'
    }

    # Configuration toggles
    PLOT_FITTED_CURVES = True     # Set to False if you only want the theoretical curves
    FIT_ALPHA = 0.5               # Transparency for fitted curves

    # --- Isolated Domain Parameters ---
    # TrmD (Domain 1):   a_n1, a_u1, m1, d1
    # Tm1570 (Domain 2): a_n2, a_u2, m2, d2

    domain_params = {
        "s1": {
            'trmd':   {'a_n': -6.46, 'a_u': -1.12, 'm': 0.94, 'd': 1.98},
            'tm1570': {'a_n': -5.64, 'a_u': -1.04, 'm': 1.33, 'd': 3.14}
        },
        "s2": {
            'trmd':   {'a_n': -5.55, 'a_u': -1.07, 'm': 1.22, 'd': 2.03},
            'tm1570': {'a_n': -4.99, 'a_u': -0.91, 'm': 1.69, 'd': 3.11}
        },
        "s3": {
            'trmd':   {'a_n': -5.04, 'a_u': -0.87, 'm': 1.17, 'd': 2.07},
            'tm1570': {'a_n': -4.60, 'a_u': -1.23, 'm': 1.86, 'd': 3.00}
        }
    }
    """
    domain_params = {
        "s1": {
            'trmd': {'a_n': -8.50, 'a_u': -1.78, 'm': 1.08, 'd': 1.93},
            'tm1570': {'a_n': -8.05, 'a_u': -1.77, 'm': 1.30, 'd': 3.17}
        },
        "s2": {
            'trmd': {'a_n': -7.81, 'a_u': -1.48, 'm': 1.08, 'd': 2.05},
            'tm1570': {'a_n': -7.29, 'a_u': -1.31, 'm': 1.19, 'd': 2.99}
        },
        "s3": {
            'trmd': {'a_n': -7.43, 'a_u': -1.22, 'm': 0.96, 'd': 2.04},
            'tm1570': {'a_n': -6.57, 'a_u': -1.20, 'm': 2.26, 'd': 3.14}
        }
    }
    """
    # Derive fixed m, d, and a_i = a_u1 + a_n2 for fitting
    fixed_params_s1 = {
        'm1': domain_params["s1"]['trmd']['m'],
        'd1': domain_params["s1"]['trmd']['d'],
        'm2': domain_params["s1"]['tm1570']['m'],
        'd2': domain_params["s1"]['tm1570']['d'],
        'a_i': domain_params["s1"]['trmd']['a_u'] + domain_params["s1"]['tm1570']['a_n']
    }

    fixed_params_s2 = {
        'm1': domain_params["s2"]['trmd']['m'],
        'd1': domain_params["s2"]['trmd']['d'],
        'm2': domain_params["s2"]['tm1570']['m'],
        'd2': domain_params["s2"]['tm1570']['d'],
        'a_i': domain_params["s2"]['trmd']['a_u'] + domain_params["s2"]['tm1570']['a_n']
    }

    fixed_params_s3 = {
        'm1': domain_params["s3"]['trmd']['m'],
        'd1': domain_params["s3"]['trmd']['d'],
        'm2': domain_params["s3"]['tm1570']['m'],
        'd2': domain_params["s3"]['tm1570']['d'],
        'a_i': domain_params["s3"]['trmd']['a_u'] + domain_params["s3"]['tm1570']['a_n']
    }

    print("Processing series1...")
    data1, popt1 = process_cd_data(path_series1, wavelength_to_check, conc_series1, series_name="series1",
                                   smoothing_method="savitzky_golay", window_size=15, poly_order=3,
                                   baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
                                   model="3-state", fixed_params=fixed_params_s1)

    print("Processing series2...")
    data2, popt2 = process_cd_data(path_series2, wavelength_to_check, conc_series2, series_name="series2",
                                   smoothing_method="savitzky_golay", window_size=15, poly_order=3,
                                   baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
                                   model="3-state", fixed_params=fixed_params_s2)

    print("Processing series3...")
    data3, popt3 = process_cd_data(path_series3, wavelength_to_check, conc_series3, series_name="series3",
                                   smoothing_method="savitzky_golay", window_size=15, poly_order=3,
                                   baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point',
                                   model="3-state", fixed_params=fixed_params_s3)

    # --- Plot Comparison for 3 Series ---
    fig_comp, ax_comp = plt.subplots(layout="constrained", figsize=(10, 7))

    series_config = [
        (data1, popt1, labels["s1"], 'blue', 'o', '-', 's1'),
        (data2, popt2, labels["s2"], 'red', 's', '--', 's2'),
        (data3, popt3, labels["s3"], 'green', '^', ':', 's3')
    ]

    x_theory = np.linspace(0, 6, 300)

    for data, popt, label, color, marker, ls, key in series_config:
        # 1. Plot experimental data points
        if data is not None and not data.empty:
            x_val = data['den_concentration'].values
            y_val = data['Ellipticity'].values
            ax_comp.scatter(x_val, y_val, label=f'{label} data', color=color, marker=marker, alpha=0.8)

            # 2. Plot optional semi-transparent fitted curves
            if PLOT_FITTED_CURVES and popt is not None:
                x_fit = np.linspace(x_val.min(), x_val.max(), 300)
                y_fit = G_three_state_weighted(x_fit, *popt)
                ax_comp.plot(x_fit, y_fit, label=f'{label} fit (a_i fixed)', color=color,
                             linestyle=ls, linewidth=2, alpha=FIT_ALPHA)

        # 3. Calculate and plot Pure Theoretical Curves (additive sum of separate domain parameters)
        trmd_p = domain_params[key]['trmd']
        tm1570_p = domain_params[key]['tm1570']

        a_n_theory = trmd_p['a_n'] + tm1570_p['a_n']
        a_i_theory = trmd_p['a_u'] + tm1570_p['a_n']
        a_u_theory = trmd_p['a_u'] + tm1570_p['a_u']

        y_theory = G_three_state_weighted(
            x_theory,
            a_n=a_n_theory,
            a_i=a_i_theory,
            a_u=a_u_theory,
            m1=trmd_p['m'], d1=trmd_p['d'],
            m2=tm1570_p['m'], d2=tm1570_p['d']
        )

        ax_comp.plot(x_theory, y_theory, label=f'{label} theoretical curve',
                     color=color, linestyle='-.', linewidth=2.5)

    ax_comp.tick_params(axis='x', labelsize=15)
    ax_comp.tick_params(axis='y', labelsize=15)
    ax_comp.set_xlabel('Denaturant concentration (M)', fontsize=16)
    ax_comp.set_ylabel('Ellipticity (mdeg)', fontsize=16)
    ax_comp.legend(fontsize=11, loc='best')
    plt.margins(0.02)

    plt.savefig(os.path.join(base_path, f"CD_{wavelength_to_check}nm_comp3_theoretical_vs_fitted_2uM.png"))
    plt.show()