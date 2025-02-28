import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(x, a, b):
    return a * x + b


def load_and_process_data(filename, columns_to_average, output_plot='combined_plot.png'):
    df = pd.read_csv(filename, sep='\t', engine='python', on_bad_lines='skip')
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    key_column = df.columns[0]
    plt.figure(figsize=(8, 5))

    for col in columns_to_average:
        if col >= df.shape[1]:
            raise ValueError(f"Invalid column index {col}. File has only {df.shape[1]} columns.")

        value_column = df.columns[col]

        def remove_outliers(group):
            mean = group.mean()
            std = group.std()
            return group[(group >= mean - 2 * std) & (group <= mean + 2 * std)]

        df_filtered = df.groupby(key_column)[value_column].apply(remove_outliers).reset_index()
        df_grouped = df_filtered.groupby(key_column)[value_column].agg(['mean', 'std']).dropna()

        df_grouped['ln_mean'] = np.log(df_grouped['mean'])
        df_grouped['ln_std'] = df_grouped['std'] / df_grouped['mean']

        df_grouped.index = pd.to_numeric(df_grouped.index, errors='coerce')
        df_grouped = df_grouped.dropna()

        popt, pcov = curve_fit(linear_fit, df_grouped.index.to_numpy(), df_grouped['ln_mean'].to_numpy())
        perr = np.sqrt(np.diag(pcov))

        print(f"Column {col}: Fitted parameters: a = {popt[0]:.4f} ± {perr[0]:.4f}, b = {popt[1]:.4f} ± {perr[1]:.4f}")

        plt.errorbar(df_grouped.index, df_grouped['ln_mean'], yerr=df_grouped['ln_std'], fmt='o',
                     label=f'Data (Col {col})')
        plt.plot(df_grouped.index, linear_fit(df_grouped.index, *popt), linestyle='-', label=f'Fit (Col {col})')

    plt.xlabel("Urea Concentration [M]")
    plt.ylabel("ln(k)")
    plt.title("Ln(k) of averaged data over urea concentration")
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()
    print(f"Plot saved as {output_plot}")


fname = "/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/Tm1570_fit_unf.txt"
output_plot = fname.rstrip("fit.txt") + "_combined_o.png"
load_and_process_data(fname, columns_to_average=[7, 9], output_plot=output_plot)