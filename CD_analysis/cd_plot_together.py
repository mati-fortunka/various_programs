import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# User-defined settings
HV_THRESHOLD = 990
SMOOTHING_METHOD = "savgol"  # Options: "savgol", "moving_average", None
SMOOTHING_WINDOW = 5        # Must be odd
SMOOTHING_POLYORDER = 3      # Only used for Savitzky-Golay
BASELINE_WAVELENGTH = 250    # Set to None to disable baseline correction

# Paths
input_folder = os.path.expanduser("/home/matifortunka/Documents/JS/kinetics_stability/biofizyka_CD/fuzja")      # Change this to your input directory
output_folder = os.path.expanduser(input_folder+"/combined")    # Change this to your output directory
range_type = ""

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def find_start_index(lines, keyword, offset=3):
    count = 0
    for i, line in enumerate(lines):
        if keyword in line:
            count += 1
            if count == 2:
                return i + offset
    raise ValueError(f"The second occurrence of '{keyword}' was not found in the file.")

def smooth_data(y):
    if SMOOTHING_METHOD == "savgol":
        if len(y) >= SMOOTHING_WINDOW:
            return savgol_filter(y, window_length=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)
    elif SMOOTHING_METHOD == "moving_average":
        if len(y) >= SMOOTHING_WINDOW:
            return np.convolve(y, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
    return y

def process_csv(file_path, hv_threshold):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[15:]

    circular_start_index = find_start_index(lines, "CircularDichroism", offset=3)
    circular_end_index = next((i for i, line in enumerate(lines[circular_start_index:], start=circular_start_index)
                               if not line.strip()), len(lines))

    hv_start_index = find_start_index(lines, "HV", offset=3)
    hv_end_index = next((i for i, line in enumerate(lines[hv_start_index:], start=hv_start_index)
                         if not line.strip()), len(lines))

    circular_lines = lines[circular_start_index:circular_end_index]
    circular_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in circular_lines])
    circular_wavelength = circular_data[:, 0]
    if circular_data.shape[1] == 2:
        circular_ellipticity_avg = circular_data[:, 1]
    elif circular_data.shape[1] > 2:
        circular_ellipticity_avg = circular_data[:, 1:].mean(axis=1)  # Average all but the first (wavelength) column
    else:
        raise ValueError(f"Unexpected number of columns in Circular Dichroism data.")

    hv_lines = lines[hv_start_index:hv_end_index]
    hv_data = np.array([list(map(float, line.replace(',', ' ').split())) for line in hv_lines])
    if hv_data.shape[1] == 2:
        hv_values = hv_data[:, 1]
    elif hv_data.shape[1] > 2:
        hv_values = hv_data[:, 1:].mean(axis=1)  # Average all but the first (wavelength) column
    else:
        raise ValueError(f"Unexpected number of columns in HV data.")

    valid_indices = hv_values <= hv_threshold
    filtered_wavelength = circular_wavelength[valid_indices]
    filtered_ellipticity_avg = circular_ellipticity_avg[valid_indices]

    sort_idx = np.argsort(filtered_wavelength)
    filtered_wavelength = filtered_wavelength[sort_idx]
    filtered_ellipticity_avg = filtered_ellipticity_avg[sort_idx]

    filtered_ellipticity_avg = smooth_data(filtered_ellipticity_avg)
    print(f"\n--- {os.path.basename(file_path)} ---")
    print(f"  Available wavelengths: {filtered_wavelength.min()}–{filtered_wavelength.max()}")
    print(f"  Ellipticity at 250nm (smoothed): {np.interp(250, filtered_wavelength, filtered_ellipticity_avg)}")
    print(f"  Ellipticity range (smoothed): {filtered_ellipticity_avg.min()} to {filtered_ellipticity_avg.max()}")

    if BASELINE_WAVELENGTH is not None:
        baseline_value = np.interp(BASELINE_WAVELENGTH, filtered_wavelength, filtered_ellipticity_avg)
        filtered_ellipticity_avg = filtered_ellipticity_avg - baseline_value
    print(f"  Ellipticity at 250nm (corrected): {np.interp(250, filtered_wavelength, filtered_ellipticity_avg)}")
    print(f"  Post-baseline range: {filtered_ellipticity_avg.min()} to {filtered_ellipticity_avg.max()}")

    return filtered_wavelength, filtered_ellipticity_avg

# Initialize plot
plt.figure(figsize=(12, 8))
plot_lines = []

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
for idx, csv_file in enumerate(csv_files):
    file_path = os.path.join(input_folder, csv_file)
    label = os.path.splitext(csv_file)[0]
    color = f"C{idx % 10}"  # Use default matplotlib color cycle

    filtered_wavelength, filtered_ellipticity_avg = process_csv(file_path, HV_THRESHOLD)
    line, = plt.plot(filtered_wavelength, filtered_ellipticity_avg, label=label, color=color)
    plot_lines.append(line)

    if "near" in csv_file.lower():
        range_type = "_nearUV"

# Finalize legend and plot
plt.legend(title="Sample Name")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Ellipticity [mdeg]')
plt.title(f'Combined Circular Dichroism Data (HV ≤ {HV_THRESHOLD} V)\nSmoothing: {SMOOTHING_METHOD or "None"}')
plt.grid(True)
plt.tight_layout()

combined_plot_path = os.path.join(output_folder, f"Combined_CD{range_type}.png")
plt.savefig(combined_plot_path)
plt.show()

print(f"Combined plot saved to: {combined_plot_path}")
