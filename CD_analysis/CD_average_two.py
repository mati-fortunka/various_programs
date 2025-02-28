import matplotlib.pyplot as plt

# File paths for the two input files
file_path_1 = '/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/fuzja_16_01/blank_ch1_036.bka'
file_path_2 = '/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/fuzja_16_01/blank_ch2_037.bka'

# Function to average data from two spectra files
def average_spectra(spectra_list):
    if len(spectra_list[0]) != len(spectra_list[1]):
        print("Error - length mismatch between files")
        return None

    average_spectra = []

    for i in range(len(spectra_list[0])):
        average_val = sum([k[i][1] for k in spectra_list]) / len(spectra_list)
        average_spectra.append([spectra_list[0][i][0], average_val])

    return average_spectra

# Function to parse a single file and extract wavelength and ellipticity data
def parse_file(file_path):
    data_start = False
    wavelengths = []
    ellipticity = []

    with open(file_path, 'r') as file:
        for line in file:
            if data_start:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wavelengths.append(float(parts[0]))
                    ellipticity.append(float(parts[1]))
            elif line.startswith('"_DATA"'):
                data_start = True

    return list(zip(wavelengths, ellipticity))

# Parse the two files
spectra_1 = parse_file(file_path_1)
spectra_2 = parse_file(file_path_2)

# Compute the average spectra
average_data = average_spectra([spectra_1, spectra_2])

# Separate the averaged data into wavelengths and ellipticity
averaged_wavelengths = [point[0] for point in average_data]
averaged_ellipticity = [point[1] for point in average_data]

# Plot the averaged ellipticity vs wavelength
plt.figure(figsize=(10, 6))
plt.plot(averaged_wavelengths, averaged_ellipticity, label="Averaged Ellipticity", color='g')
plt.title("Averaged Ellipticity vs Wavelength")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Ellipticity (mdeg)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
