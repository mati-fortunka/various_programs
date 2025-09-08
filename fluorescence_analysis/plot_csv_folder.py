import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np

def process_and_visualize_csv_with_smoothing(folder_path, output_folder, smoothing_method="moving_average",
                                             range_interval=10,  # Interval in nanometers for smoothing
                                             spline_smoothing_factor=0.5, poly_order=3):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Read the CSV file with the assumption of the provided structure
                data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", "Intensity (a.u.)"])

                # Clean the data: Convert to numeric and drop NaN rows
                data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
                data["Intensity (a.u.)"] = pd.to_numeric(data["Intensity (a.u.)"], errors="coerce")
                data.dropna(inplace=True)

                # Skip if no valid data is present
                if data.empty:
                    print(f"No valid data in {file_name}, skipping.")
                    continue

                # Determine the dynamic window size based on the range of wavelength values
                step_size = data["Wavelength (nm)"].diff().median()  # Approximate step size in the dataset
                window_size = max(3, int(np.ceil(range_interval / step_size)))  # Calculate window size in points

                # Ensure window size is odd for Savitzky-Golay filtering
                if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                    window_size += 1

                # Apply smoothing based on the chosen method
                if smoothing_method == "moving_average":
                    data["Smoothed Intensity"] = data["Intensity (a.u.)"].rolling(window=window_size, center=True).mean()
                elif smoothing_method == "spline":
                    spline = UnivariateSpline(data["Wavelength (nm)"], data["Intensity (a.u.)"],
                                              s=spline_smoothing_factor)
                    data["Smoothed Intensity"] = spline(data["Wavelength (nm)"])
                elif smoothing_method == "savitzky_golay":
                    data["Smoothed Intensity"] = savgol_filter(data["Intensity (a.u.)"], window_length=window_size, polyorder=poly_order)

                # Plot the data
                plt.figure(figsize=(10, 6))
                plt.plot(data["Wavelength (nm)"], data["Intensity (a.u.)"], label="Original")
                plt.plot(data["Wavelength (nm)"], data["Smoothed Intensity"], label="Smoothed", linestyle="--")
                plt.title(f"Spectral Data - {file_name}")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Intensity (a.u.)")
                plt.legend()
                plt.grid(True)

                # Save the plot to the output folder
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
                plt.savefig(output_file)
                plt.close()
                print(f"Visualization saved for {file_name} as {output_file}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
folder_path = ("/home/matifortunka/Documents/JS/data_Cambridge/fusions/8b1n/fluo/1")  # Replace with the path to your folder containing CSV files
output_folder = folder_path + "/output"  # Replace with the desired output folder
process_and_visualize_csv_with_smoothing(folder_path, output_folder, smoothing_method="savitzky_golay", range_interval=15, poly_order=3)
