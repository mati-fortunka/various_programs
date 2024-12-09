import os
import numpy as np
import matplotlib.pyplot as plt

# User-defined HV threshold
HV_THRESHOLD = 800  # Change this value as needed

# Path to your input folder and output folder
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Yibk_CD/"  # Replace with your folder path
output_folder = os.path.join(input_folder, "combined")
range = "_farUV"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to find the second occurrence of a keyword and return the start index
def find_start_index(lines, keyword, offset=3):
    count = 0
    for i, line in enumerate(lines):
        if keyword in line:
            count += 1
            if count == 2:
                return i + offset  # Start index is `offset` lines after the second occurrence
    raise ValueError(f"The second occurrence of '{keyword}' was not found in the file.")

# Function to process a single CSV file and return filtered data
def process_csv(file_path, hv_threshold):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[15:]

    # Find the start and end indices for Circular Dichroism data
    circular_start_index = find_start_index(lines, "CircularDichroism", offset=3)
    circular_end_index = next((i for i, line in enumerate(lines[circular_start_index:], start=circular_start_index) if not line.strip()), len(lines))

    # Find the start and end indices for HV data
    hv_start_index = find_start_index(lines, "HV", offset=3)
    hv_end_index = next((i for i, line in enumerate(lines[hv_start_index:], start=hv_start_index) if not line.strip()), len(lines))

    # Extract Circular Dichroism data
    circular_lines = lines[circular_start_index:circular_end_index]
    circular_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in circular_lines])
    circular_wavelength = circular_data[:, 0]  # First column is wavelength
    circular_ellipticity_avg = circular_data[:, -3:].mean(axis=1)  # Average of the last three columns

    # Extract HV data
    hv_lines = lines[hv_start_index:hv_end_index]
    hv_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in hv_lines])
    hv_wavelength = hv_data[:, 0]  # First column is wavelength
    hv_values = hv_data[:, -3:].mean(axis=1)  # Average of the last three columns

    # Filter data where HV is below the threshold
    valid_indices = hv_values <= hv_threshold
    filtered_wavelength = circular_wavelength[valid_indices]
    filtered_ellipticity_avg = circular_ellipticity_avg[valid_indices]

    return filtered_wavelength, filtered_ellipticity_avg

# Initialize the combined plot
plt.figure(figsize=(12, 8))

# Process all CSV files and add them to the combined plot
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    filtered_wavelength, filtered_ellipticity_avg = process_csv(file_path, HV_THRESHOLD)
    base_filename = os.path.basename(csv_file).replace(".csv", "")
    plt.plot(filtered_wavelength, filtered_ellipticity_avg, label=base_filename)
if "near" in csv_file:
    range = "_nearUV"

# Customize and save the combined plot
plt.xlabel('Wavelength [nm]')
plt.ylabel('Ellipticity [mdeg]')
plt.title(f'Combined Circular Dichroism Data (HV ≤ {HV_THRESHOLD} V)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the combined plot
combined_plot_path = os.path.join(output_folder, f"Combined_CD{range}.png")
plt.savefig(combined_plot_path)
plt.show()

print(f"Combined plot saved to: {combined_plot_path}")
