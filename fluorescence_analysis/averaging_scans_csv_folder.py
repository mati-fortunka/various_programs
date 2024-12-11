import os
import pandas as pd
import matplotlib.pyplot as plt

def process_and_visualize_averaged_scans(folder_path, save_plot=False, output="output_plot.png"):
    plt.figure(figsize=(12, 8))  # Create a figure for the combined plot

    # Initialize variables to store cumulative data and count of files
    cumulative_data = None
    scan_count = 0

    # Loop through all CSV files in the folder
    for file_name in sorted(os.listdir(folder_path)):
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

                # Increment scan count
                scan_count += 1

                # Combine data for averaging
                if cumulative_data is None:
                    cumulative_data = data.copy()
                else:
                    cumulative_data["Intensity (a.u.)"] += data["Intensity (a.u.)"].values

                # Calculate the average and plot the result
                averaged_data = cumulative_data.copy()
                averaged_data["Intensity (a.u.)"] /= scan_count

                # Plot the first scan, then the averaged scans incrementally
                label = "First Scan" if scan_count == 1 else f"Average of {scan_count} Scans"
                plt.plot(averaged_data["Wavelength (nm)"], averaged_data["Intensity (a.u.)"], label=label)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Customize the combined plot
    plt.title("Averaged Spectral Data")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()  # Add a legend for each plot
    plt.grid(True)

    # Save the plot if required
    if save_plot:
        # Ensure the output folder exists
        output_folder = f"{folder_path}/output/"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(output_folder + output)
        print(f"Plot saved to {output_folder + output}")

    # Display the plot
    plt.show()

# Example usage
folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Yibk_flourimetry/YibK_9_12/repetitions"  # Replace with the path to your folder containing CSV files
process_and_visualize_averaged_scans(folder_path, save_plot=True, output="averaged_scans_plot.png")
