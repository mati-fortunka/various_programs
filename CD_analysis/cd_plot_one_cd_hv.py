import numpy as np
import matplotlib.pyplot as plt

# Path to your input folder and CSV file
path = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/reversibility/final/cd/solenoid/solenoid 2µM native.csv"
input_folder = path.rstrip(path.split("/")[-1])  # Replace with your folder path
print(input_folder)

# Read the file
with open(path, 'r') as f:
    lines = f.readlines()
lines = lines[15:]
# Function to find the second occurrence of a keyword and return the start index
def find_start_index(lines, keyword, offset=3):
    count = 0
    for i, line in enumerate(lines):
        if keyword in line:
            count += 1
            if count == 2:
                return i + offset  # Start index is `offset` lines after the second occurrence
    raise ValueError(f"The second occurrence of '{keyword}' was not found in the file.")

# Find the start and end indices for CircularDichroism data
circular_start_index = find_start_index(lines, "CircularDichroism", offset=3)
circular_end_index = next((i for i, line in enumerate(lines[circular_start_index:], start=circular_start_index) if not line.strip()), len(lines))

# Find the start and end indices for HV data
hv_start_index = find_start_index(lines, "HV", offset=3)
hv_end_index = next((i for i, line in enumerate(lines[hv_start_index:], start=hv_start_index) if not line.strip()), len(lines))

# Extract CircularDichroism data
circular_lines = lines[circular_start_index:circular_end_index]
circular_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in circular_lines])
circular_wavelength = circular_data[:, 0]  # First column is wavelength
circular_ellipticity_avg = circular_data[:, -3:].mean(axis=1)  # Average of the last three columns

# Extract HV data
hv_lines = lines[hv_start_index:hv_end_index]
hv_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in hv_lines])
hv_wavelength = hv_data[:, 0]  # First column is wavelength
hv_values = hv_data[:, -3:].mean(axis=1)  # Average of the last three columns

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Top panel: CircularDichroism data
axs[0].plot(circular_wavelength, circular_ellipticity_avg, label='Ellipticity [mdeg]', color='blue')
axs[0].set_ylabel('Ellipticity [mdeg]')
axs[0].set_title('Circular Dichroism')
axs[0].grid(True)
axs[0].legend()

# Bottom panel: HV data
axs[1].plot(hv_wavelength, hv_values, label='High Voltage [V]', color='red')
axs[1].set_xlabel('Wavelength [nm]')
axs[1].set_ylabel('High Voltage [V]')
axs[1].set_title('HV Data')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
