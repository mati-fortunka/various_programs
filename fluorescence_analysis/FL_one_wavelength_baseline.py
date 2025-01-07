import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # RT constant kcal

# Define the function to fit
def F(x, a_n, a_u, m, g):
    return (a_n + a_u * np.exp((m * x - g) / RT)) / (1 + np.exp((m * x - g) / RT))

def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m*(x - d)) / RT)) / (1 + np.exp((m*(x - d)) / RT))

def extract_fluorescence_and_plot(folder_path, wavelength, concentration_file, smoothing_method=None, window_size=15, spline_smoothing_factor=0.5, poly_order=3, baseline_wavelength=None):
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
                #sample_number = int(file_name[:-4].split('_')[2])  # Adjust split logic as per your file naming
                sample_number = int(file_name[:-4])  # Adjust split logic as per your file naming

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

                # Dynamically determine baseline correction value
                if baseline_wavelength is not None:
                    if baseline_wavelength in data["Wavelength (nm)"].values:
                        baseline_value = data.loc[data["Wavelength (nm)"] == baseline_wavelength, "Smoothed Intensity"].values[0]
                    else:
                        # Interpolate if the exact wavelength is not available
                        baseline_value = np.interp(baseline_wavelength, data["Wavelength (nm)"], data["Smoothed Intensity"])
                else:
                    baseline_value = 0  # No baseline correction if not specified

                # Find the closest wavelength to the specified one
                closest_wavelength = data.iloc[(data["Wavelength (nm)"].sub(wavelength).abs().argsort().iloc[0])]["Wavelength (nm)"]
                target_intensity = data.loc[data["Wavelength (nm)"] == closest_wavelength, "Smoothed Intensity"].values

                if target_intensity.size > 0:
                    corrected_intensity = target_intensity[0] - baseline_value

                    # Map sample number to urea concentration
                    urea_concentration = concentration_mapping.get(sample_number, None)
                    if urea_concentration is not None:
                        fluorescence_vs_concentration.append((urea_concentration, corrected_intensity))

                    print(f"Using wavelength {closest_wavelength} nm for file {file_name}. Baseline value: {baseline_value:.2f}")
                else:
                    print(f"No valid intensity data found for file {file_name}.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert list to DataFrame for easier handling
    plot_data = pd.DataFrame(fluorescence_vs_concentration, columns=['Urea_concentration', 'Intensity'])
    plot_data.sort_values(by='Urea_concentration', inplace=True)

    # Fit the data to the function F(x)
    x_data = plot_data['Urea_concentration'].values
    y_data = plot_data['Intensity'].values

    # Initial guess for the parameters
    initial_guess = [2, 0.25, 2.5, 4.5]

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)

        # Extract the standard deviation (errors) of the parameters
        perr = np.sqrt(np.diag(pcov))

        print("Fitted parameters and their errors:")
        print(f"a_n = {popt[0]} ± {perr[0]}")
        print(f"a_u = {popt[1]} ± {perr[1]}")
        print(f"m   = {popt[2]} ± {perr[2]}")
        print(f"g (or d)   = {popt[3]} ± {perr[3]}")
    except RuntimeError as e:
        print(f"Could not fit the data: {e}")
        popt, perr = None, None

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data['Urea_concentration'], plot_data['Intensity'], label='Data', color='b', marker='o')
    if popt is not None:
        fitted_y = G(x_data, *popt)
        plt.plot(x_data, fitted_y, label='Fit', color='r', linestyle='--')
    plt.title(f'Fluorescence Intensity at {wavelength} nm vs Urea Concentration')
    plt.xlabel('Urea Concentration (M)')
    plt.ylabel('Fluorescence Intensity (a.u.)')
    plt.legend()
    plt.grid(True)
    output_file = folder_path + f"/FL_{wavelength}nm_{smoothing_method}_fit.png"
    plt.savefig(output_file)
    plt.show()

# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Tm1570_unfolding"  # Replace with the path to your folder containing CSV files
concentration_file = folder_path + "/concentrations.txt"  # Replace with the correct path to the concentrations.txt file
wavelength = 360  # Specify the wavelength of interest (e.g., 320 nm)
smoothing_method = "moving_average"  # Options: None, "moving_average", "spline", "savitzky_golay"
window_size = 20  # Used for moving average and Savitzky-Golay filter
spline_smoothing_factor = 0.5  # Used for spline smoothing
poly_order = 3  # Used for Savitzky-Golay filter
baseline_wavelength = 390  # Example baseline wavelength

extract_fluorescence_and_plot(folder_path, wavelength, concentration_file, smoothing_method, window_size, spline_smoothing_factor, poly_order, baseline_wavelength)
