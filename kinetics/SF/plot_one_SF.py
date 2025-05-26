import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_single_csv(file_path):
    """
    Reads a single CSV file, processes it, and shows the plot.
    """
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    try:
        # Read CSV without headers
        df = pd.read_csv(file_path, header=None)

        # Assign column names
        df.columns = ['time', 'voltage']

        # Filter out high voltage values
        df = df[df['voltage'] < 20]

        # Plot data
        plt.figure(figsize=(10, 5))
        plt.plot(df['time'], df['voltage'], label=os.path.basename(file_path))
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Voltage vs Time - {os.path.basename(file_path)}")
        plt.legend()
        plt.grid()
        plt.savefig(file_path.rstrip(".csv")+".png")
        plt.show()

    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage
plot_single_csv("/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/kinetics/SF/8M/3000s/8_3_zeta_3000s00026.csv")
