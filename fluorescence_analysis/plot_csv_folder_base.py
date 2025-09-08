import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np

def process_and_visualize_csv_with_smoothing(
        folder_path, output_folder, smoothing_method="moving_average",
        range_interval=10, spline_smoothing_factor=0.5, poly_order=3,
        baseline_file=None):
    """
    Process and visualize CSV spectral data with optional smoothing
    and baseline subtraction.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV spectra.
    output_folder : str
        Path to save output plots.
    smoothing_method : {"moving_average", "spline", "savitzky_golay"}, optional
        Smoothing method to apply (default "moving_average").
    range_interval : int, optional
        Interval in nanometers for smoothing window (default 10).
    spline_smoothing_factor : float, optional
        Smoothing factor for UnivariateSpline (default 0.5).
    poly_order : int, optional
        Polynomial order for Savitzky-Golay filter (default 3).
    baseline_file : str, optional
        Path to a CSV file used as baseline. If given, baseline will be
        subtracted from each spectrum.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    baseline_data = None
    if baseline_file:
        try:
            baseline_data = pd.read_csv(baseline_file, header=1, usecols=[0, 1],
                                        names=["Wavelength (nm)", "Intensity (a.u.)"])
            baseline_data = baseline_data.apply(pd.to_numeric, errors="coerce").dropna()
            baseline_data = baseline_data.sort_values("Wavelength (nm)").reset_index(drop=True)
        except Exception as e:
            print(f"Error reading baseline file {baseline_file}: {e}")
            baseline_data = None

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Skip baseline file itself
            if baseline_file and os.path.abspath(file_path) == os.path.abspath(baseline_file):
                continue

            try:
                # Read CSV
                data = pd.read_csv(file_path, header=1, usecols=[0, 1],
                                   names=["Wavelength (nm)", "Intensity (a.u.)"])
                data = data.apply(pd.to_numeric, errors="coerce").dropna()
                if data.empty:
                    print(f"No valid data in {file_name}, skipping.")
                    continue
                data = data.sort_values("Wavelength (nm)").reset_index(drop=True)

                # Subtract baseline if available
                if baseline_data is not None:
                    # Interpolate baseline to match wavelengths of current spectrum
                    baseline_interp = np.interp(
                        data["Wavelength (nm)"],
                        baseline_data["Wavelength (nm)"],
                        baseline_data["Intensity (a.u.)"]
                    )
                    data["Intensity (a.u.)"] = data["Intensity (a.u.)"] - baseline_interp

                # Determine smoothing window
                step_size = data["Wavelength (nm)"].diff().median()
                window_size = max(3, int(np.ceil(range_interval / step_size)))
                if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                    window_size += 1

                # Apply smoothing
                if smoothing_method == "moving_average":
                    data["Smoothed Intensity"] = data["Intensity (a.u.)"].rolling(window=window_size, center=True).mean()
                elif smoothing_method == "spline":
                    spline = UnivariateSpline(data["Wavelength (nm)"], data["Intensity (a.u.)"],
                                              s=spline_smoothing_factor)
                    data["Smoothed Intensity"] = spline(data["Wavelength (nm)"])
                elif smoothing_method == "savitzky_golay":
                    data["Smoothed Intensity"] = savgol_filter(
                        data["Intensity (a.u.)"],
                        window_length=window_size,
                        polyorder=poly_order
                    )

                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(data["Wavelength (nm)"], data["Intensity (a.u.)"], label="Processed")
                if "Smoothed Intensity" in data:
                    plt.plot(data["Wavelength (nm)"], data["Smoothed Intensity"], linestyle="--", label="Smoothed")
                plt.title(f"Spectral Data - {file_name}")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Intensity (a.u.)")
                plt.legend()
                plt.grid(True)

                # Save
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
                plt.savefig(output_file)
                plt.close()
                print(f"Visualization saved for {file_name} as {output_file}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# Example usage:
folder_path = "/media/matifortunka/USB-MANUAL/fluorymetry_cent/6_3_test"
output_folder = folder_path + "/output"
baseline_file = "/media/matifortunka/USB-MANUAL/fluorymetry_cent/6_3_test/EmScan1_6_3_buffer_01mm.csv"

process_and_visualize_csv_with_smoothing(
    folder_path,
    output_folder,
    smoothing_method="savitzky_golay",
    range_interval=5,
    poly_order=3,
    baseline_file=baseline_file  # <- specify baseline file
)
