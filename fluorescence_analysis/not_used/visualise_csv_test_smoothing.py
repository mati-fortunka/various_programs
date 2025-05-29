import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

def visualize_smoothing_methods(file_path, window_size=15, spline_smoothing_factor=0.5, poly_order=3):
    try:
        # Read the CSV file with the assumption of the provided structure
        data = pd.read_csv(file_path, header=1, usecols=[0, 1], names=["Wavelength (nm)", "Intensity (a.u.)"])

        # Clean the data: Convert to numeric and drop NaN rows
        data["Wavelength (nm)"] = pd.to_numeric(data["Wavelength (nm)"], errors="coerce")
        data["Intensity (a.u.)"] = pd.to_numeric(data["Intensity (a.u.)"], errors="coerce")
        data.dropna(inplace=True)

        # Skip if no valid data is present
        if data.empty:
            print(f"No valid data in {os.path.basename(file_path)}, skipping.")
            return

        # Apply smoothing methods
        data["Moving Average"] = data["Intensity (a.u.)"].rolling(window=window_size, center=True).mean()
        spline = UnivariateSpline(data["Wavelength (nm)"], data["Intensity (a.u.)"], s=spline_smoothing_factor)
        data["Spline"] = spline(data["Wavelength (nm)"])
        if window_size % 2 == 0:
            raise ValueError("Window size for Savitzky-Golay filter must be odd.")
        data["Savitzky-Golay"] = savgol_filter(data["Intensity (a.u.)"], window_length=window_size, polyorder=poly_order)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data["Wavelength (nm)"], data["Intensity (a.u.)"], label="Original", linewidth=1.5)
        plt.plot(data["Wavelength (nm)"], data["Moving Average"], label="Moving Average", linestyle="--")
        plt.plot(data["Wavelength (nm)"], data["Spline"], label="Spline", linestyle="--")
        plt.plot(data["Wavelength (nm)"], data["Savitzky-Golay"], label="Savitzky-Golay", linestyle="--")
        plt.title("Comparison of Smoothing Methods")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

# Example usage
file_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Yibk_flourimetry/YibK_unfolding/YibK_unfolding_1.csv"  # Replace with the path to your specific file
visualize_smoothing_methods(file_path, window_size=15, spline_smoothing_factor=0.5, poly_order=3)
