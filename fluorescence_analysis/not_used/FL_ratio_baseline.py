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

def G_three_state_weighted(x, a_n, a_i, a_u, m1, d1, m2, d2):
    sigmoid1 = 1 / (1 + np.exp(-(m1*(x - d1))/RT))
    sigmoid2 = 1 / (1 + np.exp(-(m2*(x - d2))/RT))
    return a_n * (1 - sigmoid1) + a_i * (sigmoid1 - sigmoid2) + a_u * sigmoid2

def guess_initial_params(x, y, model="two_state"):
    import numpy as np

    # Sort by x (concentration)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    if model == "two_state":
        # Estimate plateaus
        a_n = np.mean(y_sorted[:len(x) // 3])
        a_u = np.mean(y_sorted[-len(x) // 3:])

        # Estimate midpoint
        dy_dx = np.gradient(y_sorted, x_sorted)
        d = x_sorted[np.argmax(np.abs(dy_dx))]

        # Estimate slope
        range_y = max(y) - min(y)
        low_10 = min(y) + 0.1 * range_y
        high_90 = min(y) + 0.9 * range_y
        low_idx = np.argmin(np.abs(y - low_10))
        high_idx = np.argmin(np.abs(y - high_90))
        delta_x = abs(x[high_idx] - x[low_idx])
        m = 5 / delta_x if delta_x != 0 else 1

        return [a_n, a_u, m, d]

    elif model == "three_state":
        n = len(x_sorted)
        third = n // 3

        # Plateaus
        a_n = np.mean(y_sorted[:third])
        a_i = np.mean(y_sorted[third:2 * third])
        a_u = np.mean(y_sorted[2 * third:])

        # Midpoints of transitions
        dy_dx = np.gradient(y_sorted, x_sorted)
        transition_indices = np.argsort(np.abs(dy_dx))[-2:]
        d1, d2 = sorted(x_sorted[transition_indices])

        # Slopes
        range_y = max(y) - min(y)
        low_10 = min(y) + 0.1 * range_y
        high_90 = min(y) + 0.9 * range_y
        low_idx = np.argmin(np.abs(y - low_10))
        high_idx = np.argmin(np.abs(y - high_90))
        delta_x = abs(x[high_idx] - x[low_idx])
        m1 = m2 = 5 / delta_x if delta_x != 0 else 1

        return [a_n, a_i, a_u, m1, d1, m2, d2]

    else:
        raise ValueError("Model must be 'two_state' or 'three_state'")

def extract_fluorescence_ratio_and_plot(folder_path, wavelength1, wavelength2, concentration_file, smoothing_method=None, window_size=15, spline_smoothing_factor=0.5, poly_order=3, baseline_wavelength=None, show_fit=True):
    # Read the concentration file
    concentration_data = pd.read_csv(concentration_file, sep="\t")
    concentration_mapping = concentration_data.set_index('Sample_number')['den_concentration']

    # Prepare a list to hold data for plotting
    fluorescence_ratio_vs_concentration = []

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

                # Find the closest wavelengths to the specified ones
                closest_wavelength1 = data.iloc[(data["Wavelength (nm)"].sub(wavelength1).abs().argsort().iloc[0])]["Wavelength (nm)"]
                closest_wavelength2 = data.iloc[(data["Wavelength (nm)"].sub(wavelength2).abs().argsort().iloc[0])]["Wavelength (nm)"]

                intensity1 = data.loc[data["Wavelength (nm)"] == closest_wavelength1, "Smoothed Intensity"].values
                intensity2 = data.loc[data["Wavelength (nm)"] == closest_wavelength2, "Smoothed Intensity"].values

                if intensity1.size > 0 and intensity2.size > 0:
                    corrected_intensity1 = intensity1[0] - baseline_value
                    corrected_intensity2 = intensity2[0] - baseline_value

                    # Map sample number to denaturant concentration
                    den_concentration = concentration_mapping.get(sample_number, None)
                    if den_concentration is not None:
                        fluorescence_ratio = corrected_intensity1 / corrected_intensity2
                        fluorescence_ratio_vs_concentration.append((den_concentration, fluorescence_ratio))

                    print(f"Using wavelengths {closest_wavelength1} nm and {closest_wavelength2} nm for file {file_name}.")
                else:
                    print(f"No valid intensity data found for file {file_name}.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert list to DataFrame for easier handling
    plot_data = pd.DataFrame(fluorescence_ratio_vs_concentration, columns=['den_concentration', 'Fluorescence Ratio'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    # Fit the data to the function F(x)
    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Fluorescence Ratio'].values
    print(y_data)

    # Initial guess for the parameters
    initial_guess = guess_initial_params(x_data, y_data, model="two_state")
    initial_guess = [1.6, 0.6, 1.5, 2.7]

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)

        # Extract the standard deviation (errors) of the parameters
        perr = np.sqrt(np.diag(pcov))

        print("Fitted parameters and their errors:")
        print(f"a_n = {popt[0]} ± {perr[0]}")
        print(f"a_u = {popt[1]} ± {perr[1]}")
        print(f"m   = {popt[2]} ± {perr[2]}")
        print(f"g   = {popt[3]} ± {perr[3]}")
    except RuntimeError as e:
        print(f"Could not fit the data: {e}")
        popt, perr = None, None

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data['den_concentration'], plot_data['Fluorescence Ratio'], label='Data', color='b', marker='o')
    if popt is not None and show_fit==True:
        fitted_y = F(x_data, *popt)
        plt.plot(x_data, fitted_y, label='Fit', color='r', linestyle='--')
    plt.title(f'Fluorescence Ratio ({wavelength1} nm / {wavelength2} nm) vs Denaturant Concentration')
    plt.xlabel('Denaturant Concentration (M)')
    plt.ylabel('Fluorescence Ratio')
    plt.legend()
    plt.grid(True)
    output_file = folder_path + f"/FL_ratio_{wavelength1}_{wavelength2}_nm_fit.png"
    plt.savefig(output_file)
    plt.show()


# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/equilibrium/GdmCl/unfolding/3_rd series/WI11 _ GdHCl unf 1st set/part"  # Replace with the path to your folder containing CSV files
concentration_file = folder_path + "/concentrations.txt"  # Replace with the correct path to the concentrations.txt file
wavelength1 = 335  # Specify the first wavelength of interest (e.g., 320 nm)
wavelength2 = 355  # Specify the second wavelength of interest (e.g., 355 nm)
smoothing_method = "moving_average"  # Options: None, "moving_average", "spline", "savitzky_golay"
window_size = 15  # Used for moving average and Savitzky-Golay filter
spline_smoothing_factor = 0.5  # Used for spline smoothing
poly_order = 3  # Used for Savitzky-Golay filter
baseline_wavelength = 390  # Example baseline wavelength
fit=True

extract_fluorescence_ratio_and_plot(folder_path, wavelength1, wavelength2, concentration_file, smoothing_method, window_size, spline_smoothing_factor, poly_order, baseline_wavelength, fit)
