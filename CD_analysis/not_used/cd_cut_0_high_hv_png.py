import os
import numpy as np
import matplotlib.pyplot as plt

# User-defined HV threshold
HV_THRESHOLD = 900  # Change this value as needed

# Path to your input folder and output folder
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/equilibrium/tests/CD/point"  # Replace with your folder path
output_folder = os.path.join(input_folder, "output")

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


# Function to extract data and handle column selection
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


# Function to process a single CSV file
def process_csv(file_path, output_folder, hv_threshold):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[15:]

    # Extract Circular Dichroism data
    circular_start_index = find_start_index(lines, "CircularDichroism", offset=3)
    circular_wavelength, circular_ellipticity = extract_data(lines, circular_start_index, hv_threshold)

    # Extract HV data
    hv_start_index = find_start_index(lines, "HV", offset=3)
    hv_wavelength, hv_values = extract_data(lines, hv_start_index, hv_threshold)

    # Save raw data to a text file
    base_filename = os.path.basename(file_path).replace(".csv", "")
    raw_txt_file_path = os.path.join(output_folder, f"{base_filename}_raw.txt")
    with open(raw_txt_file_path, 'w') as txt_file:
        txt_file.write("Wavelength [nm]\tEllipticity [mdeg]\n")
        for x, y in zip(circular_wavelength, circular_ellipticity):
            txt_file.write(f"{x}\t{y}\n")

    # Filter data where HV is below the threshold
    indices = hv_values > hv_threshold
    circular_ellipticity[indices] = 0

    # Save filtered data to a text file
    filtered_txt_file_path = os.path.join(output_folder, f"{base_filename}_t.txt")
    with open(filtered_txt_file_path, 'w') as txt_file:
        txt_file.write("Wavelength [nm]\tEllipticity [mdeg]\n")
        for x, y in zip(circular_wavelength, circular_ellipticity):
            txt_file.write(f"{x}\t{y}\n")

    # Plot filtered Circular Dichroism data
    plt.figure(figsize=(10, 6))
    plt.plot(circular_wavelength, circular_ellipticity, label='Ellipticity [mdeg]', color='blue')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Ellipticity [mdeg]')
    plt.title(f'Circular Dichroism (HV ≤ {hv_threshold} V)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PNG file
    png_file_path = os.path.join(output_folder, f"{base_filename}_t.png")
    plt.savefig(png_file_path)
    plt.close()

    print(f"Processed: {file_path}")
    print(f"  Raw data saved to: {raw_txt_file_path}")
    print(f"  Truncated data saved to: {filtered_txt_file_path}")
    print(f"  Plot saved to: {png_file_path}")


# Process all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
for csv_file in csv_files:
    process_csv(os.path.join(input_folder, csv_file), output_folder, HV_THRESHOLD)

print(f"Processing completed. All outputs are saved in: {output_folder}")
