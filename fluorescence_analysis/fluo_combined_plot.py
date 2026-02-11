import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np
from matplotlib.colors import Normalize
import re


def process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average",
                                                  range_interval=10,
                                                  spline_smoothing_factor=0.5, poly_order=3,
                                                  baseline_wavelength=400,
                                                  data_type="Fluorescence",  # Options: "CD" or "Fluorescence"
                                                  save_plot=True, output="output_plot.png"):
    # --- CONFIGURATION BASED ON DATA TYPE ---
    if data_type == "CD":
        y_col_name = "Ellipticity (mdeg)"
        y_axis_label = "Ellipticity (mdeg)"
    else:
        # Assuming generic names for fluorimetry, change "Intensity" if your CSV header is different
        y_col_name = "Ellipticity (mdeg)"  # <--- CHECK THIS: Does your fluorescence CSV really say 'Ellipticity'?
        y_axis_label = "Fluorescence (a.u.)"

    fig, ax = plt.subplots(figsize=(12, 8))

    # Load the concentration data
    concentrations_path = os.path.join(folder_path, "concentrations.txt")
    try:
        concentrations_data = pd.read_csv(concentrations_path, sep="\t", header=0)
        concentration_map = dict(zip(concentrations_data.iloc[:, 0], concentrations_data.iloc[:, 1]))
    except Exception as e:
        print(f"Error reading concentrations.txt: {e}")
        return

    # --- REGEX FILE SORTING (Your correct implementation) ---
    def extract_number(filename):
        match = re.findall(r'(\d+)', filename)
        return int(match[-1]) if match else None

    csv_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.csv'):
            num = extract_number(f)
            if num is not None:
                csv_files.append((f, num))

    csv_files.sort(key=lambda x: x[1])

    plot_data = []

    for file_name, file_number in csv_files:
        denaturant_concentration = round(concentration_map.get(file_number, float('nan')), 2)

        if np.isnan(denaturant_concentration):
            print(f"No concentration found for file {file_name} (ID: {file_number}), skipping.")
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            # Note: Verify strictly if your CSV header matches y_col_name defined above
            data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", y_col_name])
            data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
            data[y_col_name] = pd.to_numeric(data[y_col_name], errors="coerce")

            # 1. Initial cleanup
            data.dropna(inplace=True)

            if data.empty:
                print(f"No valid data in {file_name}, skipping.")
                continue

            # Determine window size
            step_size = data["Wavelength (nm)"].diff().median()
            window_size = max(3, int(np.ceil(range_interval / step_size)))

            if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                window_size += 1

            # --- SMOOTHING ---
            if smoothing_method == "moving_average":
                data["Smoothed"] = data[y_col_name].rolling(window=window_size, center=True).mean()
            elif smoothing_method == "spline":
                spline = UnivariateSpline(data["Wavelength (nm)"], data[y_col_name], s=spline_smoothing_factor)
                data["Smoothed"] = spline(data["Wavelength (nm)"])
            elif smoothing_method == "savitzky_golay":
                data["Smoothed"] = savgol_filter(data[y_col_name], window_length=window_size, polyorder=poly_order)
            else:
                data["Smoothed"] = data[y_col_name]

            # 2. CRITICAL: Drop NaNs created by smoothing BEFORE baseline correction
            data.dropna(subset=["Smoothed"], inplace=True)

            # --- BASELINE CORRECTION ---
            if baseline_wavelength:
                # Find index closest to baseline wavelength
                # idxmin() is safe here because we dropped NaNs
                baseline_idx = (np.abs(data["Wavelength (nm)"] - baseline_wavelength)).idxmin()

                # Check if the closest wavelength is actually close (within 5nm) to avoid grabbing far-off data
                closest_wv = data.loc[baseline_idx, "Wavelength (nm)"]
                if abs(closest_wv - baseline_wavelength) > 5:
                    print(
                        f"Warning: Nearest data point to baseline ({closest_wv}nm) is far from target ({baseline_wavelength}nm) in {file_name}")

                baseline_value = data.loc[baseline_idx, "Smoothed"]
                data["Smoothed"] -= baseline_value

            plot_data.append((denaturant_concentration, data["Wavelength (nm)"], data["Smoothed"]))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Sort and Plot
    plot_data.sort(key=lambda x: x[0])

    if not plot_data:
        print("No data was processed successfully.")
        return

    concentrations = [item[0] for item in plot_data]
    cmap = plt.colormaps['viridis']
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    colors = [cmap(norm(c)) for c in concentrations]

    for (concentration, wavelength, smoothed), color in zip(plot_data, colors):
        ax.plot(wavelength, smoothed, color=color, linewidth=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Denaturant Concentration (M)")

    title_text = f"Combined Data ({data_type})"
    if baseline_wavelength:
        title_text += f" - Baseline Corrected at {baseline_wavelength} nm"

    ax.set_title(title_text)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(y_axis_label)
    ax.grid(True, linestyle='--', alpha=0.7)

    if save_plot:
        output_folder = os.path.join(folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output)
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")

    plt.show()


# Example usage
folder_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/fluorimetry/TrmD"

# Note: I added data_type="Fluorescence".
# Ensure your CSV columns are actually named "Ellipticity (mdeg)".
# If they are named "Intensity" or similar, change line 23 in the function.
process_and_visualize_combined_with_smoothing(
    folder_path,
    smoothing_method="savitzky_golay",
    range_interval=10,
    poly_order=3,
    save_plot=True,
    output="output_plot.png",
    baseline_wavelength=400,  # Set to e.g., 360 or 400 if needed
    data_type="Fluorescence"
)