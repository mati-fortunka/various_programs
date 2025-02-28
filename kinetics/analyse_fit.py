import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(x, a, b):
    return a * x + b


def load_and_process_data(filename, column_to_average=2, output_plot='averaged_plot.png'):
    bad_rows = []

    # Load the file with error handling
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            columns = line.strip().split('\t')
            if len(columns) < column_to_average + 1:
                bad_rows.append((i + 1, line.strip()))

    # Print bad rows
    if bad_rows:
        print("Bad rows detected:")
        for row in bad_rows:
            print(f"Line {row[0]}: {row[1]}")

    df = pd.read_csv(filename, sep='\t', engine='python', on_bad_lines='skip')

    # Print shape to check how many rows are loaded
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Ensure column_to_average is within range
    if column_to_average >= df.shape[1]:
        raise ValueError(f"Invalid column index {column_to_average}. File has only {df.shape[1]} columns.")

    # Group by first column (urea concentration) and compute mean and std deviation for selected column
    grouped = df.groupby(df.columns[0])[df.columns[column_to_average]].agg(['mean', 'std'])
    grouped['ln_mean'] = np.log(grouped['mean'])
    grouped['ln_std'] = grouped['std'] / grouped['mean']

    # Fit data to a linear curve
    popt, pcov = curve_fit(linear_fit, grouped.index.to_numpy(dtype=float), grouped['ln_mean'].to_numpy(dtype=float))
    perr = np.sqrt(np.diag(pcov))

    print(f"Fitted parameters: a = {popt[0]:.4f} ± {perr[0]:.4f}, b = {popt[1]:.4f} ± {perr[1]:.4f}")

    # Plot results with error bars
    plt.figure(figsize=(8, 5))
    plt.errorbar(grouped.index, grouped['ln_mean'], yerr=grouped['ln_std'], fmt='o', label='Data')
    plt.plot(grouped.index, linear_fit(grouped.index, *popt), linestyle='-',
             label=f'Fit: y = {popt[0]:.4f}x + {popt[1]:.4f}')
    plt.xlabel("Urea Concentration [M]")
    plt.ylabel("ln(k)")
    plt.title("Ln(k) of averaged data over urea concentration")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")


# Example usage
fname = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Tm1570_fit.txt"

out = fname.rstrip("fit.txt") + "_full_double_fast"
load_and_process_data(fname, column_to_average=7, output_plot=out)

out = fname.rstrip("fit.txt") + "_full_slope"
load_and_process_data(fname, column_to_average=4, output_plot=out)

out = fname.rstrip("fit.txt") + "_full_double_slow"
load_and_process_data(fname, column_to_average=9, output_plot=out)
