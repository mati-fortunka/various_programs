import os
import matplotlib.pyplot as plt

# Define the folder containing .bka files and the output folder
input_folder = '/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/fuzja_16_01'  # Current directory
output_folder = input_folder + '/output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def process_bka_file(file_path):
    """Extracts data from a .bka file and returns wavelengths and ellipticity."""
    data_start = False
    wavelengths = []
    ellipticity = []

    with open(file_path, 'r') as file:
        for line in file:
            if data_start:
                # Split the line into columns and convert to float
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wavelengths.append(float(parts[0]))
                    ellipticity.append(float(parts[1]))
            elif line.startswith('"_DATA"'):
                # Start reading data after the "_DATA" line
                data_start = True

    return wavelengths, ellipticity

# Process all .bka files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.bka'):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

        # Extract data from the file
        wavelengths, ellipticity = process_bka_file(file_path)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, ellipticity, label="Ellipticity", color='b')
        plt.title(f"Ellipticity vs Wavelength: {filename}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Ellipticity (mdeg)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory

        print(f"Processed and saved plot for: {filename}")

print("All .bka files have been processed and saved in the 'output' folder.")
