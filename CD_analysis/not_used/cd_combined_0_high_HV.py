import os
import numpy as np
import matplotlib.pyplot as plt

# User-defined HV threshold
HV_THRESHOLD = 700  # Change this value as needed

# Path to your input folder and output folder
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/22h/CD/fuzja_F8_unfolding_spectra_widmo_JS_26_04"  # Replace with your folder path
output_folder = os.path.join(input_folder, "combined")
# range = "_farUV"
range = ""
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


# Function to extract data and handle HV threshold
def extract_data(lines, start_index, hv_threshold, columns_to_check=3):
    end_index = next((i for i, line in enumerate(lines[start_index:], start=start_index) if not line.strip()),
                     len(lines))
    data_lines = lines[start_index:end_index]
    data = np.array([list(map(float, line.replace(',', ' ').split())) for line in data_lines])

    # Check if there are enough columns to average
    num_columns = data.shape[1]
    if num_columns >= columns_to_check:
        values = data[:, -columns_to_check:].mean(axis=1)  # Average of the last columns
    else:
        values = data[:, -1]  # Use the last column if not enough columns

    return data[:, 0], values  # Return wavelengths and values


# Function to process a single CSV file and return filtered data
def process_csv(file_path, hv_threshold):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[15:]

    # Extract Circular Dichroism data
    circular_start_index = find_start_index(lines, "CircularDichroism", offset=3)
    circular_wavelength, circular_ellipticity = extract_data(lines, circular_start_index, hv_threshold)

    # Extract HV data
    hv_start_index = find_start_index(lines, "HV", offset=3)
    _, hv_values = extract_data(lines, hv_start_index, hv_threshold)

    # Set ellipticity to 0 where HV exceeds the threshold
    circular_ellipticity[hv_values > hv_threshold] = 0

    return circular_wavelength, circular_ellipticity


# Initialize the combined plot
plt.figure(figsize=(12, 8))

# Process all CSV files and add them to the combined plot
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    circular_wavelength, circular_ellipticity = process_csv(file_path, HV_THRESHOLD)
    base_filename = os.path.basename(csv_file).replace(".csv", "")
    plt.plot(circular_wavelength, circular_ellipticity, label=base_filename)
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
