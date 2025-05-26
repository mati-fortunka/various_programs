import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(folder_path, truncate_start_time=0.0, truncate_end_time=0.0):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    output_folder = os.path.join(folder_path, 'plots')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            try:
                # Read CSV without headers
                df_full = pd.read_csv(file_path, header=None, names=['time', 'voltage'])

                # Detect wrap: when time suddenly decreases
                wrap_index = None
                time_col = df_full['time'].values
                for i in range(1, len(time_col)):
                    if time_col[i] <= time_col[i - 1]:
                        wrap_index = i
                        break

                # Truncate at wrap point if found
                if wrap_index is not None:
                    print(f"Detected time wrap at row {wrap_index}. Truncating data.")
                    df = df_full.iloc[:wrap_index]
                else:
                    df = df_full

                # Drop duplicate time values, keeping the first
                df = df.drop_duplicates(subset='time', keep='first')

                # Filter out high voltage values
                df = df[df['voltage'] < 20].reset_index(drop=True)

                # Apply start/end truncation based on time
                df = df[(df['time'] >= df['time'].min() + truncate_start_time)]
                df = df[(df['time'] <= df['time'].max() - truncate_end_time)]

                # Plot the truncated segment
                if len(df) > 1:
                    plt.figure(figsize=(10, 5))
                    plt.plot(df['time'], df['voltage'], label=filename)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Voltage (V")
                    plt.title(f"Voltage vs Time - {filename}")
                    plt.legend()
                    plt.grid()

                    output_file = os.path.join(output_folder, filename.replace('.csv', '_segment1.png'))
                    plt.savefig(output_file)
                    plt.close()
                    print(f"Saved plot: {output_file}")
                else:
                    print(f"Skipped plot for {filename}: Not enough data after truncation.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
plot_csv_files(
    "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Z/kinetics/SF/6M",
    truncate_start_time=0,  # seconds to cut from the start
    truncate_end_time=0    # seconds to cut from the end
)
