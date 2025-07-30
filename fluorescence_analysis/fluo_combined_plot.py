import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np
from matplotlib.colors import Normalize

def process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average",
                                                  range_interval=10,  # Interval in nanometers for smoothing
                                                  spline_smoothing_factor=0.5, poly_order=3, baseline_wavelength = 390,
                                                  save_plot=True, output="output_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis for the combined plot

    # Load the concentration data
    concentrations_path = os.path.join(folder_path, "concentrations.txt")
    try:
        concentrations_data = pd.read_csv(concentrations_path, sep="\t", header=0)
        concentration_map = dict(zip(concentrations_data.iloc[:, 0], concentrations_data.iloc[:, 1]))
    except Exception as e:
        print(f"Error reading concentrations.txt: {e}")
        return

    # Collect the files and sort them by numerical order
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    plot_data = []  # To hold all processed data for ordered plotting

    # Loop through all sorted CSV files
    for file_name in csv_files:
        file_number = int(os.path.splitext(file_name)[0])  # Extract the number from the filename
        denaturant_concentration = round(concentration_map.get(file_number, float('nan')), 2)

        if np.isnan(denaturant_concentration):
            print(f"No concentration found for file {file_name}, skipping.")
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", "Ellipticity (mdeg)"])
            data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
            data["Ellipticity (mdeg)"] = pd.to_numeric(data["Ellipticity (mdeg)"], errors="coerce")
            data.dropna(inplace=True)

            if data.empty:
                print(f"No valid data in {file_name}, skipping.")
                continue

            wavelength_range = data["Wavelength (nm)"].max() - data["Wavelength (nm)"].min()
            step_size = data["Wavelength (nm)"].diff().median()
            window_size = max(3, int(np.ceil(range_interval / step_size)))

            if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                window_size += 1

            # Smoothing
            if smoothing_method == "moving_average":
                data["Smoothed Ellipticity"] = data["Ellipticity (mdeg)"].rolling(window=window_size, center=True).mean()
            elif smoothing_method == "spline":
                spline = UnivariateSpline(data["Wavelength (nm)"], data["Ellipticity (mdeg)"], s=spline_smoothing_factor)
                data["Smoothed Ellipticity"] = spline(data["Wavelength (nm)"])
            elif smoothing_method == "savitzky_golay":
                data["Smoothed Ellipticity"] = savgol_filter(data["Ellipticity (mdeg)"],
                                                             window_length=window_size,
                                                             polyorder=poly_order)
            else:
                data["Smoothed Ellipticity"] = data["Ellipticity (mdeg)"]

            if baseline_wavelength:
                # Baseline correction at baseline_wavelength
                baseline_index = (np.abs(data["Wavelength (nm)"] - baseline_wavelength)).idxmin()
                baseline_value = data.loc[baseline_index, "Smoothed Ellipticity"]
                data["Smoothed Ellipticity"] -= baseline_value

            # Store for plotting
            plot_data.append((denaturant_concentration, data["Wavelength (nm)"], data["Smoothed Ellipticity"]))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Sort the plot data by concentration
    plot_data.sort(key=lambda x: x[0])

    # Generate color map
    concentrations = [item[0] for item in plot_data]
    cmap = plt.colormaps['viridis']
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    colors = [cmap(norm(c)) for c in concentrations]

    for (concentration, wavelength, smoothed), color in zip(plot_data, colors):
        ax.plot(wavelength, smoothed, color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Denaturant Concentration (M)")

    ax.set_title("Combined Spectral Data with Smoothing and Baseline Correction at 390 nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Ellipticity (mdeg)")
    ax.grid(True)

    if save_plot:
        output_folder = os.path.join(folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    plt.show()


# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/22h/fluo/22h/spectra_83_F4N_JS_26_04"  # Replace with your path
process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average", range_interval=15,
                                              poly_order=3, save_plot=True, output="output_plot.png", baseline_wavelength = None)
