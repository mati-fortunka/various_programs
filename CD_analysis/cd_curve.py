import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # RT constant kcal
scans = 1

# 2-state model function
def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


# Helper to find second occurrence of a keyword
def find_start_index(lines, keyword, offset=3):
    count = 0
    for i, line in enumerate(lines):
        if keyword in line:
            count += 1
            if count == 2:
                return i + offset
    raise ValueError(f"The second occurrence of '{keyword}' was not found in the file.")


# Main function
def extract_cd_and_plot(folder_path, wavelength, concentration_file, smoothing_method=None,
                        window_size=15, spline_smoothing_factor=0.5, poly_order=3,
                        baseline_wavelength=None):
    # Load concentration file
    concentration_data = pd.read_csv(concentration_file, sep="\t")
    concentration_mapping = concentration_data.set_index('Sample_number')['den_concentration']

    ellipticity_vs_concentration = []
    all_spectra = []  # <<< ADDED: Store spectra for combined plot >>>

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                lines = lines[15:]

                cd_start = find_start_index(lines, "CircularDichroism", offset=3)
                cd_end = next((i for i, line in enumerate(lines[cd_start:], start=cd_start) if not line.strip()),
                              len(lines))
                cd_lines = lines[cd_start:cd_end]

                cd_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in cd_lines])
                wavelengths = cd_data[:, 0]
                ellipticity = cd_data[:, -scans:].mean(axis=1)

                if smoothing_method == "moving_average":
                    smoothed = pd.Series(ellipticity).rolling(window=window_size, center=True).mean().to_numpy()
                elif smoothing_method == "spline":
                    spline = UnivariateSpline(wavelengths, ellipticity, s=spline_smoothing_factor)
                    smoothed = spline(wavelengths)
                elif smoothing_method == "savitzky_golay":
                    if window_size % 2 == 0:
                        raise ValueError("Window size for Savitzky-Golay filter must be odd.")
                    smoothed = savgol_filter(ellipticity, window_length=window_size, polyorder=poly_order)
                else:
                    smoothed = ellipticity

                if wavelengths[0] > wavelengths[-1]:
                    wavelengths = wavelengths[::-1]
                    smoothed = smoothed[::-1]

                if baseline_wavelength is not None:
                    baseline_value = np.interp(baseline_wavelength, wavelengths, smoothed)
                else:
                    baseline_value = 0

                closest_wavelength = wavelengths[np.argmin(np.abs(wavelengths - wavelength))]
                target_ellipticity = smoothed[np.argmin(np.abs(wavelengths - wavelength))]
                corrected_ellipticity = target_ellipticity - baseline_value

                file_pattern = re.compile(r"(\d{5})\.csv$")
                match = file_pattern.search(file_name)
                sample_number = int(match.group(1))

                den_conc = concentration_mapping.get(sample_number, None)
                if den_conc is not None:
                    ellipticity_vs_concentration.append((den_conc, corrected_ellipticity))
                    all_spectra.append((wavelengths, smoothed, f"{den_conc:.2f} M"))  # <<< ADDED

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Plot ellipticity vs concentration
    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Ellipticity'].values

    initial_guess = [min(y_data), max(y_data), 2, 2]
    try:
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        print("Fitted parameters and their errors:")
        print(f"a_n = {popt[0]} ± {perr[0]}")
        print(f"a_u = {popt[1]} ± {perr[1]}")
        print(f"m   = {popt[2]} ± {perr[2]}")
        print(f"d   = {popt[3]} ± {perr[3]}")
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        popt = None

    # Plotting fit
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label='Data', color='blue', marker='o')
    if popt is not None:
        plt.plot(x_data, G(x_data, *popt), label='Fit', color='red', linestyle='--')
    plt.title(f'Ellipticity at {wavelength} nm vs Denaturant Concentration')
    plt.xlabel('Denaturant Concentration (M)')
    plt.ylabel('Mean Ellipticity (mdeg)')
    plt.grid(True)
    plt.legend()
    output_file = folder_path + f"/CD_{wavelength}nm_{smoothing_method}_fit.png"
    plt.savefig(output_file)
    plt.show()

    # <<< ADDITION: Plot all spectra >>>
    plt.figure(figsize=(10, 7))
    for wls, sm, label in all_spectra:
        plt.plot(wls, sm, label=label)
    plt.title("Smoothed CD Spectra for All Samples")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Ellipticity (mdeg)")
    plt.grid(True)
    plt.legend(title="Denaturant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    combined_plot_file = os.path.join(folder_path, "combined_cd_spectra.png")
    plt.savefig(combined_plot_file)
    plt.show()

    return plot_data, popt



# Example usage
path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/22h/CD/fuzja_F8_unfolding_spectra_widmo_JS_26_04"
concentrations = path + "/concentrations.txt"

extract_cd_and_plot(path, 220, concentrations, smoothing_method="savitzky_golay",
                    window_size=15, spline_smoothing_factor=0.5, poly_order=3,
                    baseline_wavelength=250)
