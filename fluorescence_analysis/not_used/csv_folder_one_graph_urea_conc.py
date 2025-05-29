import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np


def process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average",
                                                  range_interval=10,  # Interval in nanometers for smoothing
                                                  spline_smoothing_factor=0.5, poly_order=3,
                                                  save_plot=False, output="output_plot.png"):
    plt.figure(figsize=(12, 8))  # Create a figure for the combined plot

    # Load the concentration data
    concentrations_path = os.path.join(folder_path, "concentrations.txt")
    concentration_map = {}
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

    # Loop through all sorted CSV files
    for file_name in csv_files:
        file_number = int(os.path.splitext(file_name)[0])  # Extract the number from the filename
        urea_concentration = round(concentration_map.get(file_number, float('nan')), 2)  # Get and round the concentration

        if np.isnan(urea_concentration):
            print(f"No concentration found for file {file_name}, skipping.")
            continue

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

            # Add the smoothed line to the combined plot
            plt.plot(data["Wavelength (nm)"], data["Smoothed Ellipticity"],
                     label=f"{urea_concentration} M Urea ({smoothing_method})")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Customize the combined plot
    plt.title("Combined Spectral Data with Smoothing")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Ellipticity (mdeg)")
    plt.legend(title="Denaturant Concentration")  # Add a legend with a title
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
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/equilibrium/GdmCl/unfolding/2nd_series/fluo/6_3 WI11 unf GdHCl after 16h 2 uM"  # Replace with the path to your folder containing CSV files
process_and_visualize_combined_with_smoothing(folder_path, smoothing_method="moving_average", range_interval=15,
                                              poly_order=3, save_plot=True, output="output_plot.png")
