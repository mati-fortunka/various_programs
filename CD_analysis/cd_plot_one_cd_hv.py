import numpy as np
import matplotlib.pyplot as plt

# Path to your input folder and CSV file
path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/1st_set/CD/F8E4N00001.csv"
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


# Helper function to parse and process data
def extract_data(lines, keyword):
    start_idx = find_start_index(lines, keyword, offset=3)
    end_idx = next((i for i, line in enumerate(lines[start_idx:], start=start_idx) if not line.strip()), len(lines))
    data_lines = lines[start_idx:end_idx]
    data_array = np.array([list(map(float, line.replace(',', ' ').split())) for line in data_lines])

    if data_array.shape[1] == 2:
        values = data_array[:, 1]
    elif data_array.shape[1] > 2:
        values = data_array[:, 1:].mean(axis=1)  # Average all but the first (wavelength) column
    else:
        raise ValueError(f"Unexpected number of columns in {keyword} data.")

    return data_array[:, 0], values


# Extract CircularDichroism and HV data
circular_wavelength, circular_ellipticity_avg = extract_data(lines, "CircularDichroism")
hv_wavelength, hv_values = extract_data(lines, "HV")

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
