import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data from CSV
data = np.loadtxt('dane_trmd.csv', delimiter=',', unpack=True)

# Define the degradation function
def f(x, V, Km):
    return V * x / (x + Km)

# Initial parameters
V_init = 0.9
Km_init = 4.1
Vmax = 0.99

# Fit the data
x_data = data[0]
y_data = data[1]
yerr_data = data[2]
popt, pcov = curve_fit(f, x_data, y_data, p0=[V_init, Km_init], sigma=yerr_data)

# Extract fitted parameters and their errors
V_fit, Km_fit = popt
V_err, Km_err = np.sqrt(np.diag(pcov))

# Print final Km, V_max, and their errors
print(f"Final Km: {Km_fit} ± {Km_err}")
print(f"Final V_max: {V_fit} ± {V_err}")

# Create plot
plt.figure(figsize=(8, 6))

# Plot data with error bars
plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='o', label="Data", color="#0000EE", lw=2, capsize=5)

# Plot fitted curve
x_fit = np.linspace(0, 60, 500)
y_fit = f(x_fit, V_fit, Km_fit)
plt.plot(x_fit, y_fit, label="Fit", color="#000000", lw=2)

# Plot horizontal and vertical lines for Km, Vmax/2
plt.axhline(y=Vmax/2, color="#E61C66", linestyle='--', lw=1.5)
plt.axvline(x=Km_fit, color="#E61C66", linestyle='--', lw=1.5)

# Plot Vmax line
plt.axhline(y=Vmax, color="#E61C66", linestyle='--', lw=1.5)

# Set titles and labels
plt.title('Degradation')
plt.xlabel('[TrmD-ssra] (uM)')
plt.ylabel('Degradation rate (uM/min)')
plt.xlim(0, 60)
plt.ylim(0, 1.2)

# Show legend in top right
plt.legend(loc='upper right')

# Show plot
plt.show()
