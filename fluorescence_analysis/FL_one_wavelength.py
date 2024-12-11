import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

def extract_fluorescence_and_plot(folder_path, wavelength, concentration_file, smoothing_method=None, window_size=15, spline_smoothing_factor=0.5, poly_order=3):
    # Read the concentration file
    concentration_data = pd.read_csv(concentration_file, sep="\t")
    concentration_mapping = concentration_data.set_index('Sample_number')['Urea_concentration']

    # Prepare a list to hold data for plotting
    fluorescence_vs_concentration = []

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Extract the sample number from the file name
                sample_number = int(file_name[:-4].split('_')[2])  # Adjust split logic as per your file naming

                # Read the CSV file, assuming it has wavelength and intensity columns
                data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", "Intensity (a.u.)"])

                # Clean the data
                data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
                data["Intensity (a.u.)"] = pd.to_numeric(data["Intensity (a.u.)"], errors="coerce")
                data.dropna(inplace=True)

                # Apply smoothing if specified
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
                else:
                    data["Smoothed Intensity"] = data["Intensity (a.u.)"]  # No smoothing

                # Find the closest wavelength to the specified one
                closest_wavelength = data.iloc[(data["Wavelength (nm)"].sub(wavelength).abs().argsort().iloc[0])]["Wavelength (nm)"]
                target_intensity = data.loc[data["Wavelength (nm)"] == closest_wavelength, "Smoothed Intensity"].values

                if target_intensity.size > 0:
                    # Map sample number to urea concentration
                    urea_concentration = concentration_mapping.get(sample_number, None)
                    if urea_concentration is not None:
                        fluorescence_vs_concentration.append((urea_concentration, target_intensity[0]))

                    print(f"Using wavelength {closest_wavelength} nm for file {file_name}.")
                else:
                    print(f"No valid intensity data found for file {file_name}.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert list to DataFrame for easier handling
    plot_data = pd.DataFrame(fluorescence_vs_concentration, columns=['Urea_concentration', 'Intensity'])
    plot_data.sort_values(by='Urea_concentration', inplace=True)

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(plot_data['Urea_concentration'], plot_data['Intensity'], marker='o', linestyle='-', color='b')
    plt.title(f'Fluorescence Intensity at {wavelength} nm vs Urea Concentration')
    plt.xlabel('Urea Concentration (M)')
    plt.ylabel('Fluorescence Intensity (a.u.)')
    plt.grid(True)
    output_file = folder_path + f"/FL_{wavelength}nm_{smoothing_method}.png"
    plt.savefig(output_file)
    plt.show()

# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Yibk_flourimetry/YibK_unfolding"  # Replace with the path to your folder containing CSV files
concentration_file = folder_path + "/concentrations.txt"  # Replace with the correct path to the concentrations.txt file
wavelength = 355  # Specify the wavelength of interest (e.g., 320 nm)
smoothing_method = "moving_average"  # Options: None, "moving_average", "spline", "savitzky_golay"
window_size = 15  # Used for moving average and Savitzky-Golay filter
spline_smoothing_factor = 0.5  # Used for spline smoothing
poly_order = 3  # Used for Savitzky-Golay filter

extract_fluorescence_and_plot(folder_path, wavelength, concentration_file, smoothing_method, window_size, spline_smoothing_factor, poly_order)
