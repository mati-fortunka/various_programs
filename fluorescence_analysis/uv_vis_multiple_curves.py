import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/my_proteins/uv_vis/proteins.csv'
data = pd.read_csv(file_path, header=None)  # Read the CSV without assuming a header

# Extract sample names and axis labels
sample_names = data.iloc[0]  # First row: Sample names
axis_labels = data.iloc[1]  # Second row: Axis labels (e.g., Wavelength and Absorbance)

# Initialize storage for curve data
curve_data = []

# Iterate through every pair of columns
for i in range(0, len(data.columns), 2):
    if i + 1 >= len(data.columns):
        break  # Ensure there is a pair of columns

    # Extract sample name and axis labels
    sample_name = sample_names[i]
    x_label = axis_labels[i]
    y_label = axis_labels[i + 1]

    # Check if any required information is missing
    if pd.isna(sample_name) or pd.isna(x_label) or pd.isna(y_label):
        break  # Stop processing if a required field is empty

    # Verify data integrity by attempting conversion to numeric
    try:
        x_data = data[i][2:].astype(float)
        y_data = data[i + 1][2:].astype(float)
    except ValueError:
        continue  # Skip this pair if data cannot be converted to float

    # Append the processed data for plotting
    curve_data.append((sample_name, x_label, y_label, x_data, y_data))

# Plot the curves
plt.figure(figsize=(12, 8))

for sample_name, x_label, y_label, x_data, y_data in curve_data:
    plt.plot(x_data, y_data, label=sample_name)

# Enhance the plot appearance
plt.title("UV-Vis Spectra", fontsize=16)
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Absorbance", fontsize=14)
plt.legend(title="Samples", fontsize=10, loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
