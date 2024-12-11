import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

def process_and_visualize_combined_with_3d_smoothing(folder_path, smoothing_method="moving_average", window_size=15,
                                                      spline_smoothing_factor=0.5, poly_order=3):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Extract the third axis value from the last two characters of the file name (excluding .csv)
                z_value = int(file_name[-6:-4])  # Adjust based on the filename structure

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

                # Apply smoothing based on the chosen method
                if smoothing_method == "moving_average":
                    data["Smoothed Intensity"] = data["Intensity (a.u.)"].rolling(window=window_size, center=True).mean()
                elif smoothing_method == "spline":
                    spline = UnivariateSpline(data["Wavelength (nm)"], data["Intensity (a.u.)"],
                                              s=spline_smoothing_factor)
                    data["Smoothed Intensity"] = spline(data["Wavelength (nm)"])
                elif smoothing_method == "savitzky_golay":
                    if window_size % 2 == 0:
                        raise ValueError("Window size for Savitzky-Golay filter must be odd.")
                    data["Smoothed Intensity"] = savgol_filter(data["Intensity (a.u.)"], window_length=window_size, polyorder=poly_order)

                # Add the smoothed line to the 3D plot
                ax.plot(data["Wavelength (nm)"], [z_value] * len(data), data["Smoothed Intensity"], label=f"{file_name} ({smoothing_method})")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Customize the 3D plot
    ax.set_title("3D Combined Spectral Data with Smoothing")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("File Identifier (Z-Axis)")
    ax.set_zlabel("Intensity (a.u.)")
    #ax.legend()  # Add a legend for each file
    plt.grid(True)

    # Display the plot
    plt.show()

# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Yibk_flourimetry/YibK_unfolding/"  # Replace with the path to your folder containing CSV files
process_and_visualize_combined_with_3d_smoothing(folder_path, smoothing_method="moving_average", window_size=15, poly_order=3)
