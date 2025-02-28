import matplotlib.pyplot as plt

# Define the file path
file_path = '/home/matifortunka/Documents/JS/CD/Mateusz_Tm1570/unfolding/0m_ch1_008.bka'

# Parse the file to extract wavelength and ellipticity data
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

# Plot the ellipticity (mdeg) vs wavelength (nm)
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, ellipticity, label="Ellipticity", color='b')
plt.title("Ellipticity vs Wavelength")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Ellipticity (mdeg)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
