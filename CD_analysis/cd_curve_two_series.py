import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # RT constant kcal
scans = 1


# 2-state model function
def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


def G_three_state_weighted(x, a_n, a_i, a_u, m1, d1, m2, d2):
    sigmoid1 = 1 / (1 + np.exp(-(m1 * (x - d1)) / RT))
    sigmoid2 = 1 / (1 + np.exp(-(m2 * (x - d2)) / RT))
    return a_n * (1 - sigmoid1) + a_i * (sigmoid1 - sigmoid2) + a_u * sigmoid2


# Helper: find the data block for a given property
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
                    poly_order=3, baseline_wavelength=None, hv_cutoff=700, hv_mode='per_point'):
    concentration_data = pd.read_csv(concentration_file, sep="\t")
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
                if den_conc is not None:
                    ellipticity_vs_concentration.append((den_conc, corrected_ellipticity))
                    all_spectra.append((wavelengths, smoothed, den_conc))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not ellipticity_vs_concentration:
        raise RuntimeError(f"No data collected for {series_name}.")

    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Ellipticity'].values

    # Fit curve
    initial_guess = [min(y_data), max(y_data), 2, 2]
    try:
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        print(f"\n--- Fitted parameters for {series_name} ---")
        print(f"a_n = {popt[0]:.4f} ± {perr[0]:.4f}")
        print(f"a_u = {popt[1]:.4f} ± {perr[1]:.4f}")
        print(f"m   = {popt[2]:.4f} ± {perr[2]:.4f}")
        print(f"d   = {popt[3]:.4f} ± {perr[3]:.4f}")
        print("-" * 35)
    except Exception as e:
        print(f"Curve fitting failed for {series_name}: {e}")
        popt = None

    # Heatbar-style combined spectra plot (per folder)
    if all_spectra:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        denaturant_values = [c for _, _, c in all_spectra]
        vmin, vmax = min(denaturant_values), max(denaturant_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.viridis

        for wls, sm, conc in all_spectra:
            ax2.plot(wls, sm, color=cmap(norm(conc)), alpha=0.9)

        smap = cm.ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        cbar = fig2.colorbar(smap, ax=ax2)
        cbar.set_label("Denaturant concentration (M)")

        ax2.set_title(f"Smoothed CD Spectra - {series_name}")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Ellipticity (mdeg)")
        ax2.grid(True)
        fig2.tight_layout()
        combined_plot_file = os.path.join(folder_path, f"{series_name}_spectra_heatbar.png")
        fig2.savefig(combined_plot_file)
        plt.close(fig2)

    return plot_data, popt


# Example usage
if __name__ == "__main__":

    # Define paths for both series (Update these with your actual local paths)
    base_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/trmd/"

    path_series1 = os.path.join(base_path, "seria2_1")
    conc_series1 = os.path.join(path_series1, "concentrations.txt")

    # path_series2 = os.path.join(base_path, "seria2_1", "best")  # Replace with actual series2 path
    path_series2 = os.path.join(base_path, "seria2_2")  # Replace with actual series2 path

    conc_series2 = os.path.join(path_series2, "concentrations.txt")

    wavelength_to_check = 217

    print("Processing series1...")
    data1, popt1 = process_cd_data(path_series1, wavelength_to_check, conc_series1, series_name="series1",
                                   smoothing_method="savitzky_golay", window_size=15, poly_order=3,
                                   baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point')

    print("Processing series2...")
    data2, popt2 = process_cd_data(path_series2, wavelength_to_check, conc_series2, series_name="series2",
                                   smoothing_method="savitzky_golay", window_size=15, poly_order=3,
                                   baseline_wavelength=250, hv_cutoff=990, hv_mode='per_point')

    # --- Plot Comparison ---
    fig_comp, ax_comp = plt.subplots(figsize=(10, 7))

    # Plot Series 1
    if data1 is not None and not data1.empty:
        x1 = data1['den_concentration'].values
        y1 = data1['Ellipticity'].values
        ax_comp.scatter(x1, y1, label='series1 Data', color='blue', marker='o')
        if popt1 is not None:
            x_fit1 = np.linspace(x1.min(), x1.max(), 200)
            ax_comp.plot(x_fit1, G(x_fit1, *popt1), label='series1 Fit', color='blue', linestyle='-')

    # Plot Series 2
    if data2 is not None and not data2.empty:
        x2 = data2['den_concentration'].values
        y2 = data2['Ellipticity'].values
        ax_comp.scatter(x2, y2, label='series2 Data', color='red', marker='s')
        if popt2 is not None:
            x_fit2 = np.linspace(x2.min(), x2.max(), 200)
            ax_comp.plot(x_fit2, G(x_fit2, *popt2), label='series2 Fit', color='red', linestyle='--')

    ax_comp.set_title(f'Ellipticity Comparison at {wavelength_to_check} nm vs Denaturant Concentration')
    ax_comp.set_xlabel('Denaturant Concentration (M)')
    ax_comp.set_ylabel('Mean Ellipticity (mdeg)')
    ax_comp.grid(True, linestyle=':', alpha=0.7)
    ax_comp.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"CD_{wavelength_to_check}nm_comparison_fit.png"))
    plt.show()