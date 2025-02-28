import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_files(folder_path):
    """
    Reads all CSV files from the given folder, processes them, and saves the plots as PNG files.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Ensure output directory exists
    output_folder = os.path.join(folder_path, 'plots')
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            try:
                # Read CSV without headers
                df = pd.read_csv(file_path, header=None)

                # Assign column names
                df.columns = ['time', 'voltage']

                # Filter out high voltage values
                df = df[df['voltage'] <= 1000]

                # Plot data
                plt.figure(figsize=(10, 5))
                plt.plot(df['time'], df['voltage'], label=filename)
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.title(f"Voltage vs Time - {filename}")
                plt.legend()
                plt.grid()

                # Save plot
                output_file = os.path.join(output_folder, filename.replace('.csv', '.png'))
                plt.savefig(output_file)
                plt.close()

                print(f"Saved plot: {output_file}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
plot_csv_files("/home/matifortunka/Documents/JS/data_Cambridge/MateuszF/SF/to_fit/unf/2uM")
