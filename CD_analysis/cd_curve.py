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

# Helper: find the data block for a given property (e.g. "CircularDichroism" or "HV")
def find_block(lines, property_name):
    """
    Returns (start_index, end_index) of the data lines for the block whose header equals property_name.
    Expectation in file:
      Wavelength,
      <PropertyName>
      <data lines...>
      (possibly) blank line or next "Wavelength," starting next block
    """
    for i in range(len(lines) - 1):
        # look for line that equals "Wavelength," and the next line containing the property name
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith(property_name):
            start = i + 2  # data begins after the property line
            # find end: blank line or next "Wavelength," or end of file
            end = start
            while end < len(lines) and lines[end].strip():
                # stop if the next block header begins
                if lines[end].strip().startswith("Wavelength"):
                    break
                end += 1
            return start, end
    return None, None


# Main function
def extract_cd_and_plot(folder_path, wavelength, concentration_file, smoothing_method=None,
                        window_size=15, spline_smoothing_factor=0.5, poly_order=3,
                        baseline_wavelength=None, hv_cutoff=700, hv_mode='per_point'):
    """
    hv_mode:
      - 'per_point'   : remove wavelengths where HV > hv_cutoff (default)
      - 'per_spectrum': discard entire file if any HV > hv_cutoff
      - None or False : do no HV filtering
    """
    # Load concentration file
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

            # Many files have a header — skip top lines that are not part of data blocks.
            # Heuristic: find first "Wavelength" occurrence and keep from a bit before it.
            # But to be safe, just pass full file to find_block.
            # Extract CD block
            cd_start, cd_end = find_block(lines, "CircularDichroism")
            if cd_start is None:
                raise ValueError(f"CircularDichroism block not found in {file_name}")
            cd_lines = [ln.strip() for ln in lines[cd_start:cd_end] if ln.strip()]
            # parse numeric pairs
            cd_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in cd_lines]
            cd_data = np.array(cd_pairs)
            if cd_data.shape[1] < 2:
                raise ValueError(f"Unexpected CD data shape in {file_name}: {cd_data.shape}")

            wavelengths_cd = cd_data[:, 0]
            ellipticity = cd_data[:, 1]

            # Extract HV block (if present)
            hv_start, hv_end = find_block(lines, "HV")
            if hv_start is None:
                # No HV block: treat as no HV (don't filter)
                hv_values = np.full_like(wavelengths_cd, fill_value=np.nan, dtype=float)
                print(f"Warning: HV block not found in {file_name}; skipping HV filtering for this file.")
            else:
                hv_lines = [ln.strip() for ln in lines[hv_start:hv_end] if ln.strip()]
                hv_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in hv_lines]
                hv_data = np.array(hv_pairs)
                if hv_data.shape[1] < 2:
                    raise ValueError(f"Unexpected HV data shape in {file_name}: {hv_data.shape}")
                wavelengths_hv = hv_data[:, 0]
                hv_values = hv_data[:, 1]

                # Align wavelengths (allow a tolerance if needed)
                if wavelengths_cd.shape != wavelengths_hv.shape or not np.allclose(wavelengths_cd, wavelengths_hv):
                    # try to align by intersection (robust)
                    common, idx_cd, idx_hv = np.intersect1d(wavelengths_cd, wavelengths_hv, return_indices=True)
                    if common.size == 0:
                        raise ValueError(f"No matching wavelengths between CD and HV in {file_name}")
                    wavelengths_cd = wavelengths_cd[idx_cd]
                    ellipticity = ellipticity[idx_cd]
                    hv_values = hv_values[idx_hv]

            # Apply HV filtering modes
            if hv_cutoff is not None and hv_mode:
                if hv_mode == 'per_point':
                    mask = np.ones_like(hv_values, dtype=bool)
                    # If hv_values contains NaN (no HV block), keep them
                    mask = np.where(np.isnan(hv_values), True, hv_values <= hv_cutoff)
                    # apply mask
                    wavelengths = wavelengths_cd[mask]
                    ellipticity = ellipticity[mask]
                    hv_values = hv_values[mask]
                elif hv_mode == 'per_spectrum':
                    # if any hv > cutoff then skip entire spectrum
                    if np.any(~np.isnan(hv_values) & (hv_values > hv_cutoff)):
                        print(f"Skipping {file_name} because HV exceeded cutoff ({hv_cutoff}) in per_spectrum mode.")
                        continue
                    else:
                        wavelengths = wavelengths_cd
                else:
                    raise ValueError("hv_mode must be 'per_point', 'per_spectrum', or None/False")
            else:
                wavelengths = wavelengths_cd

            # If nothing remains after filtering, skip file
            if wavelengths.size == 0 or ellipticity.size == 0:
                print(f"Skipping {file_name}: no data after HV filtering.")
                continue

            # Smoothing: only if enough points for chosen method
            smoothed = ellipticity.copy()
            if smoothing_method == "moving_average":
                if len(ellipticity) >= window_size:
                    smoothed = pd.Series(ellipticity).rolling(window=window_size, center=True).mean().to_numpy()
                    # fallback: replace NaNs at edges with original values
                    nan_idx = np.isnan(smoothed)
                    smoothed[nan_idx] = ellipticity[nan_idx]
                else:
                    print(f"Not enough points for moving_average for {file_name}; skipping smoothing.")

            elif smoothing_method == "spline":
                # need at least (order+1) points, UnivariateSpline can handle small amounts but we'll require >=4
                if len(wavelengths) >= 4:
                    spline = UnivariateSpline(wavelengths, ellipticity, s=spline_smoothing_factor)
                    smoothed = spline(wavelengths)
                else:
                    print(f"Not enough points for spline smoothing for {file_name}; skipping smoothing.")

            elif smoothing_method == "savitzky_golay":
                if window_size % 2 == 0:
                    raise ValueError("Window size for Savitzky-Golay filter must be odd.")
                if len(ellipticity) >= window_size:
                    smoothed = savgol_filter(ellipticity, window_length=window_size, polyorder=poly_order)
                else:
                    print(f"Not enough points for savitzky_golay for {file_name}; skipping smoothing.")
            else:
                smoothed = ellipticity

            # Ensure wavelength ascending
            if wavelengths[0] > wavelengths[-1]:
                wavelengths = wavelengths[::-1]
                smoothed = smoothed[::-1]

            # Baseline correction (interpolate baseline_wavelength)
            if baseline_wavelength is not None:
                # if baseline wavelength outside measured range, np.interp will extrapolate; that's OK but warn
                if baseline_wavelength < wavelengths.min() or baseline_wavelength > wavelengths.max():
                    print(f"Warning: baseline_wavelength {baseline_wavelength} out of range for {file_name}")
                baseline_value = np.interp(baseline_wavelength, wavelengths, smoothed)
            else:
                baseline_value = 0.0

            # get ellipticity at requested wavelength (interpolate if wavelength not present)
            target_ellipticity = np.interp(wavelength, wavelengths, smoothed)
            corrected_ellipticity = target_ellipticity - baseline_value

            # extract sample number from filename (if matching your naming scheme)
            file_pattern = re.compile(r"(\d{5})\.csv$")
            match = file_pattern.search(file_name)
            if not match:
                print(f"Warning: file name {file_name} doesn't match expected pattern; skipping concentration mapping.")
                sample_number = None
            else:
                sample_number = int(match.group(1))

            den_conc = concentration_mapping.get(sample_number, None) if sample_number is not None else None
            if den_conc is not None:
                ellipticity_vs_concentration.append((den_conc, corrected_ellipticity))
                all_spectra.append((wavelengths, smoothed, den_conc))
            else:
                # If no concentration mapping found, skip adding to main result but still keep spectra if desired.
                print(f"Note: sample {sample_number} from {file_name} not found in concentration file; skipping concentration entry.")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Prepare plot data
    if not ellipticity_vs_concentration:
        raise RuntimeError("No data collected (ellipticity_vs_concentration is empty). Check files and concentration mapping.")

    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Ellipticity'].values

    # Fit curve
    initial_guess = [min(y_data), max(y_data), 2, 2]
    try:
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        print("Fitted parameters and their errors:")
        print(f"a_n = {popt[0]} ± {perr[0]}")
        print(f"a_u = {popt[1]} ± {perr[1]}")
        print(f"m   = {popt[2]} ± {perr[2]}")
        print(f"d   = {popt[3]} ± {perr[3]}")
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        popt = None

    # Plotting fit
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(x_data, y_data, label='Data', color='blue', marker='o')
    if popt is not None:
        ax1.plot(np.linspace(x_data.min(), x_data.max(), 200), G(np.linspace(x_data.min(), x_data.max(), 200), *popt),
                 label='Fit', color='red', linestyle='--')
    ax1.set_title(f'Ellipticity at {wavelength} nm vs Denaturant Concentration')
    ax1.set_xlabel('Denaturant Concentration (M)')
    ax1.set_ylabel('Mean Ellipticity (mdeg)')
    ax1.grid(True)
    ax1.legend()
    output_file = os.path.join(folder_path, f"CD_{wavelength}nm_{smoothing_method}_fit.png")
    fig1.savefig(output_file)
    plt.show()

    # Heatbar-style combined spectra plot
    if all_spectra:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        denaturant_values = [c for _, _, c in all_spectra]
        vmin = min(denaturant_values)
        vmax = max(denaturant_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.viridis

        for wls, sm, conc in all_spectra:
            ax2.plot(wls, sm, color=cmap(norm(conc)), alpha=0.9)

        smap = cm.ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])  # for compatibility
        cbar = fig2.colorbar(smap, ax=ax2)
        cbar.set_label("Denaturant concentration (M)")

        ax2.set_title("Smoothed CD Spectra (Heatbar by Denaturant Concentration)")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Ellipticity (mdeg)")
        ax2.grid(True)
        fig2.tight_layout()
        combined_plot_file = os.path.join(folder_path, "combined_cd_spectra_heatbar.png")
        fig2.savefig(combined_plot_file)
        plt.show()
    else:
        print("No spectra to plot in combined plot (all_spectra is empty).")

    return plot_data, popt


# Example usage
if __name__ == "__main__":
    path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/22h/CD/fuzja_F8_unfolding_spectra_widmo_JS_26_04"
    concentrations = os.path.join(path, "concentrations.txt")

    # hv_mode: 'per_point' (remove points where HV>cutoff) OR 'per_spectrum' (skip file if any HV>cutoff)
    extract_cd_and_plot(path, 220, concentrations, smoothing_method="savitzky_golay",
                        window_size=15, spline_smoothing_factor=0.5, poly_order=3,
                        baseline_wavelength=250, hv_cutoff=800, hv_mode='per_point')
