import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np
import io


def process_and_visualize_csv_with_smoothing(folder_path, output_folder, smoothing_method="moving_average",
                                             range_interval=15,
                                             spline_smoothing_factor=0.5, poly_order=3,
                                             x_label="Wavelength (nm)", y_label="Intensity (a.u.)",
                                             show_raw_data=True):  # <-- NEW TOGGLE ADDED HERE
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # 1) TRULY BULLETPROOF DATA LOADING (Stops after first data block)
                valid_data_lines = []
                in_data_block = False

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split(',')

                        if len(parts) >= 2:
                            try:
                                # Try to parse as numbers
                                x = float(parts[0])
                                y = float(parts[1])
                                valid_data_lines.append(f"{x},{y}\n")
                                in_data_block = True
                            except ValueError:
                                # If it's text but we were already reading data, the block is over.
                                if in_data_block:
                                    break
                                continue
                        else:
                            # If it's an empty line or single column after data started, the block is over.
                            if in_data_block:
                                break

                # Skip if no valid data was found
                if not valid_data_lines:
                    print(f"No valid numeric data found in {file_name}, skipping.")
                    continue

                # Load the perfectly clean data block into pandas
                data = pd.read_csv(io.StringIO("".join(valid_data_lines)), names=["X", "Y"])

                # Determine the dynamic window size based on the range of X values
                step_size = data["X"].diff().median()

                # Prevent division by zero if step_size is 0 or NaN
                if pd.isna(step_size) or step_size == 0:
                    step_size = 1.0

                window_size = max(3, int(np.ceil(range_interval / step_size)))

                # Ensure window size is odd for Savitzky-Golay filtering
                if smoothing_method == "savitzky_golay" and window_size % 2 == 0:
                    window_size += 1
                # Savitzky-Golay requires the window size to be strictly greater than poly_order
                if smoothing_method == "savitzky_golay" and window_size <= poly_order:
                    window_size = poly_order + 2 if (poly_order + 2) % 2 != 0 else poly_order + 3

                # Apply smoothing based on the chosen method
                if smoothing_method == "moving_average":
                    data["Smoothed Y"] = data["Y"].rolling(window=window_size, center=True).mean()
                elif smoothing_method == "spline":
                    # Spline needs X strictly increasing, so we sort it just in case
                    data = data.sort_values(by="X")
                    spline = UnivariateSpline(data["X"], data["Y"], s=spline_smoothing_factor)
                    data["Smoothed Y"] = spline(data["X"])
                elif smoothing_method == "savitzky_golay":
                    data["Smoothed Y"] = savgol_filter(data["Y"], window_length=window_size, polyorder=poly_order)

                # 2) IMPROVED PLOTTING WITH TOGGLE
                fig, ax = plt.subplots(figsize=(10, 6))

                # Conditionally plot the raw data based on the toggle
                if show_raw_data:
                    ax.plot(data["X"], data["Y"], label="Original", alpha=0.6)  # alpha makes raw data slightly lighter
                    smoothed_linestyle = "--"
                else:
                    smoothed_linestyle = "-"  # Use a solid line if it's the only one plotted

                # Always plot the smoothed data
                ax.plot(data["X"], data["Smoothed Y"], label="Smoothed", linestyle=smoothed_linestyle, color='black',
                        linewidth=2)

                # Apply labels passed into the function
                ax.set_xlabel(x_label, fontsize=16)
                ax.set_ylabel(y_label, fontsize=16)
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.margins(0.02)

                # Only show legend if we are plotting both lines
                if show_raw_data:
                    ax.legend(fontsize=15)

                # ax.grid(True)

                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
                print(f"Visualization saved for {file_name} as {output_file}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# Example usage for your Kinetics (Time) data
folder_path = ("/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/TrmD/seria2_1")
output_folder = folder_path + "/output"

process_and_visualize_csv_with_smoothing(
    folder_path,
    output_folder,
    smoothing_method="savitzky_golay",
    range_interval=25,
    poly_order=3,
    x_label="Wavelength (nm)",
    y_label="Counts",
    show_raw_data=False  # CHANGE THIS TO True IF YOU WANT TO SEE BOTH LINES
)