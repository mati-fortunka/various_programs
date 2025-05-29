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

def extract_fluorescence_and_plot(folder_path, concentration_file, smoothing_method=None, window_size=15, spline_smoothing_factor=0.5, poly_order=3):
    # Read the concentration file
    concentration_data = pd.read_csv(concentration_file, sep="\t")
    concentration_mapping = concentration_data.set_index('Sample_number')['Urea_concentration']

    # Prepare a list to hold data for plotting
    ew_vs_concentration = []

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

                # Calculate the equivalent width (EW)
                ew = (data["Smoothed Intensity"] * data["Wavelength (nm)"].values).sum()
                # Map sample number to urea concentration
                urea_concentration = concentration_mapping.get(sample_number, None)
                if urea_concentration is not None:
                    ew_vs_concentration.append((urea_concentration, ew))

                print(f"Processed file {file_name}, EW={ew:.2f}.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert list to DataFrame for easier handling
    plot_data = pd.DataFrame(ew_vs_concentration, columns=['Urea_concentration', 'EW'])
    plot_data.sort_values(by='Urea_concentration', inplace=True)

    # Fit the data to the function F(x)
    x_data = plot_data['Urea_concentration'].values
    y_data = plot_data['EW'].values

    # Initial guess for the parameters
    initial_guess = [35000, 15000, 3.5, 4.5]

    try:
        # Perform curve fitting
        #popt, pcov = curve_fit(F, x_data, y_data, p0=initial_guess)
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
    plt.scatter(plot_data['Urea_concentration'], plot_data['EW'], label='Data', color='b', marker='o')
    if popt is not None:
        fitted_y = G(x_data, *popt)
        plt.plot(x_data, fitted_y, label='Fit', color='r', linestyle='--')
    plt.title('Equivalent Width (EW) vs Urea Concentration')
    plt.xlabel('Urea Concentration (M)')
    plt.ylabel('Equivalent Width (a.u.)')
    plt.legend()
    plt.grid(True)
    output_file = folder_path + f"/EW_vs_urea_{smoothing_method}_fit.png"
    plt.savefig(output_file)
    plt.show()


# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Tm1570_unfolding"  # Replace with the path to your folder containing CSV files
concentration_file = folder_path + "/concentrations.txt"  # Replace with the correct path to the concentrations.txt file
smoothing_method = "moving_average"  # Options: None, "moving_average", "spline", "savitzky_golay"
window_size = 20  # Used for moving average and Savitzky-Golay filter
spline_smoothing_factor = 0.5  # Used for spline smoothing
poly_order = 3  # Used for Savitzky-Golay filter

extract_fluorescence_and_plot(folder_path, concentration_file, smoothing_method, window_size, spline_smoothing_factor, poly_order)
