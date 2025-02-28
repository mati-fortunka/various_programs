import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np


def process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average",
                                                  range_interval=10,  # Interval in nanometers for smoothing
                                                  spline_smoothing_factor=0.5, poly_order=3,
                                                  baseline_wavelength=390,  # Baseline wavelength for subtraction
                                                  save_plot=False, output="output_plot.png"):
    plt.figure(figsize=(12, 8))  # Create a figure for the combined plot

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Read the CSV file with the assumption of the provided structure
                data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", "Ellipticity (mdeg)"])

                # Clean the data: Convert to numeric and drop NaN rows
                data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
                data["Ellipticity (mdeg)"] = pd.to_numeric(data["Ellipticity (mdeg)"], errors="coerce")
                data.dropna(inplace=True)

                # Skip if no valid data is present
                if data.empty:
                    print(f"No valid data in {file_name}, skipping.")
                    continue

                # Determine the dynamic window size based on the range of wavelength values
                wavelength_range = data["Wavelength (nm)"].max() - data["Wavelength (nm)"].min()
                step_size = data["Wavelength (nm)"].diff().median()  # Approximate step size in the dataset
                window_size = max(3, int(np.ceil(range_interval / step_size)))  # Calculate window size in points

                # Ensure window size is odd for Savitzky-Golay filtering
                if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                    window_size += 1

                # Apply smoothing based on the chosen method
                if smoothing_method == "moving_average":
                    data["Smoothed Ellipticity"] = data["Ellipticity (mdeg)"].rolling(window=window_size,
                                                                                      center=True).mean()
                elif smoothing_method == "spline":
                    spline = UnivariateSpline(data["Wavelength (nm)"], data["Ellipticity (mdeg)"],
                                              s=spline_smoothing_factor)
                    data["Smoothed Ellipticity"] = spline(data["Wavelength (nm)"])
                elif smoothing_method == "savitzky_golay":
                    data["Smoothed Ellipticity"] = savgol_filter(data["Ellipticity (mdeg)"], window_length=window_size,
                                                                 polyorder=poly_order)
                else:
                    data["Smoothed Ellipticity"] = data["Ellipticity (mdeg)"]

                # Determine the baseline value at the specified wavelength
                closest_index = (data["Wavelength (nm)"] - baseline_wavelength).abs().idxmin()
                baseline_value = data.loc[closest_index, "Smoothed Ellipticity"]

                # Subtract the baseline from the smoothed data
                data["Baseline Corrected"] = data["Smoothed Ellipticity"] - baseline_value

                # Add the baseline-corrected line to the combined plot
                plt.plot(data["Wavelength (nm)"], data["Baseline Corrected"],
                         label=f"{file_name} ({smoothing_method})")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Customize the combined plot
    plt.title("Combined Spectral Data with Dynamic Baseline Correction")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Baseline-Corrected Ellipticity (mdeg)")
    plt.legend()  # Add a legend for each file
    plt.grid(True)

    # Save the plot if required
    if save_plot:
        # Ensure the output folder exists
        output_folder = os.path.join(folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    # Display the plot
    plt.show()


# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/slow_phase_test/to_compare/2"  # Replace with the path to your folder containing CSV files
process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average", range_interval=15,
                                              poly_order=3, baseline_wavelength=390, save_plot=True,
                                              output="output_plot.png")
