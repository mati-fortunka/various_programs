import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob
from io import StringIO
from matplotlib.colors import to_rgb


def read_data(file_content_or_path):
    """
    Reads data from a CSV file, handling 'ProDataCSV' and standard CSV formats.
    It extracts the wavelength label, processes time wraps, and cleans the data.

    Args:
        file_content_or_path (str): Either a file path or the raw content of the CSV file.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The processed DataFrame with 'Time' and 'CircularDichroism' data.
            - str: The extracted wavelength label (e.g., "Wavelength: 222 nm").
    """
    is_file_path = os.path.exists(file_content_or_path) if isinstance(file_content_or_path, str) else False

    if is_file_path:
        with open(file_content_or_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = file_content_or_path.splitlines()

    wavelength_label = "wavelength (222 nm)"
    header_found = False

    # Check for ProDataCSV format
    if lines[0].strip() == "ProDataCSV":
        print(f"🧠 ProDataCSV structure detected.")
        data_block_start = -1
        # Find start of the CircularDichroism data section
        for i, line in enumerate(lines):
            if "Wavelength,Wavelength:" in line:
                match = re.search(r'Wavelength:\s*(\d+(?:\.\d+)?)\s*nm', line)
                if match:
                    wavelength_label = f"Wavelength: {match.group(1)} nm"
            if line.strip() == "CircularDichroism":
                data_block_start = i
            if data_block_start != -1 and line.strip().startswith("Time") and "Wavelength" in line:
                header_found = True
                df_full = pd.read_csv(StringIO('\n'.join(lines[i + 1:])), sep=",", header=None)
                break
        else:
            raise ValueError("❌ Could not locate 'CircularDichroism' data header (e.g., 'Time,Wavelength').")

        # Assuming first two columns are Time and CircularDichroism
        df_full = df_full.iloc[:, :2].copy()
        df_full.rename(columns={0: 'Time', 1: 'CircularDichroism'}, inplace=True)

    else:
        # Standard CSV format
        print(f"📄 Standard CSV structure detected.")
        # Try to infer header and wavelength
        first_line = lines[0].strip()
        skiprows = 0
        if not re.match(r'^-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?', first_line.split(',')[0]):
            match = re.search(r'(\d+(?:\.\d+)?)', first_line)
            if match:
                wavelength_label = f"Wavelength: {match.group(1)} nm"
            else:
                wavelength_label = first_line
            skiprows = 1

        df_full = pd.read_csv(StringIO('\n'.join(lines)), skiprows=skiprows, sep=",")
        df_full = df_full.dropna(how='all', axis=1)
        # Assuming the first two columns are Time and CircularDichroism
        if df_full.shape[1] < 2:
            raise ValueError("File does not have at least two columns for 'Time' and 'CircularDichroism'.")
        df_full = df_full.iloc[:, :2].copy()
        df_full.columns = ['Time', 'CircularDichroism']
        header_found = True

    if not header_found:
        raise ValueError("❌ Failed to parse data from file.")

    # Convert to numeric and handle potential errors
    df_full['Time'] = pd.to_numeric(df_full['Time'], errors='coerce')
    df_full['CircularDichroism'] = pd.to_numeric(df_full['CircularDichroism'], errors='coerce')
    df_full = df_full.dropna()

    # Handle time wrap
    time_col = df_full['Time'].values
    wrap_index = np.where(np.diff(time_col) <= 0)[0]
    if wrap_index.size > 0:
        first_wrap_idx = wrap_index[0] + 1
        print(f"⚠️ Detected time wrap at row {first_wrap_idx}. Truncating data.")
        df = df_full.iloc[:first_wrap_idx].copy()
    else:
        df = df_full.copy()

    df = df.drop_duplicates(subset=['Time'], keep='first')
    return df, wavelength_label


def moving_average(data, window_size):
    """Applies a simple moving average to the data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def exponential(t, A, k, c):
    """Single exponential decay function."""
    return A * np.exp(-k * t) + c


def single_exponential_with_drift(t, A, k, c, m):
    """Single exponential decay with linear drift."""
    return A * np.exp(-k * t) + c + m * t


def double_exponential(t, A1, k1, A2, k2, c):
    """Double exponential decay function."""
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def estimate_initial_k(time, intensity):
    """Estimates an initial rate constant (k) for exponential fitting."""
    time_series = pd.Series(time)
    intensity_series = pd.Series(intensity)
    A0 = intensity_series.iloc[0] - intensity_series.iloc[-1]
    C = intensity_series.iloc[-1]
    half_decay_val = C + A0 / 2
    # Find the index where the intensity is closest to half_decay_val
    half_decay_index = np.abs(intensity_series - half_decay_val).argmin()
    t_half = time_series.iloc[half_decay_index]
    k_init = np.log(2) / t_half if t_half > 0 else 0.01
    return k_init


def fit_data(time, intensity, fit_type, dead_time):
    """
    Fits data to a specified kinetic model.

    Returns:
        tuple: (params, params_std_dev, fit_function, fit_summary_string, t_half_value, ratio_a1)
    """
    time = time + dead_time
    fit_start = time.iloc[0]
    fit_end = time.iloc[-1]
    mask = (time >= fit_start) & (time <= fit_end)
    fit_time = time[mask]
    fit_cd = intensity[mask]

    params, pcov, fit_func, fit_summary, t_half, ratio_a1 = None, None, None, "Fit failed.", None, None

    try:
        if fit_type == 'exponential':
            A0 = fit_cd.iloc[0] - fit_cd.iloc[-1]
            C = fit_cd.iloc[-1]
            k0 = estimate_initial_k(fit_time, fit_cd)
            initial_guess = [A0, k0, C]
            params, pcov = curve_fit(exponential, fit_time, fit_cd, p0=initial_guess, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            fit_summary = (
                f"Fit params: A={params[0]:.3f}±{perr[0]:.3f}, k={params[1]:.3f}±{perr[1]:.3f} (t_half={np.log(2) / params[1]:.3f}s), c={params[2]:.3f}±{perr[2]:.3f}")
            fit_func = exponential
            t_half = np.log(2) / params[1]
        elif fit_type == 'single_exponential_with_drift':
            A0 = fit_cd.iloc[0] - fit_cd.iloc[-1]
            C = fit_cd.iloc[-1]
            k0 = estimate_initial_k(fit_time, fit_cd)
            m0 = (fit_cd.iloc[-1] - fit_cd.iloc[0]) / (fit_time.iloc[-1] - fit_time.iloc[0])  # Initial guess for drift
            initial_guess = [A0, k0, C, m0]
            params, pcov = curve_fit(single_exponential_with_drift, fit_time, fit_cd, p0=initial_guess, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            fit_summary = (
                f"Fit params: A={params[0]:.3f}±{perr[0]:.3f}, k={params[1]:.3f}±{perr[1]:.3f} (t_half={np.log(2) / params[1]:.3f}s), c={params[2]:.3f}±{perr[2]:.3f}, m={params[3]:.3f}±{perr[3]:.3f}")
            fit_func = single_exponential_with_drift
            t_half = np.log(2) / params[1]
        elif fit_type == 'double_exponential':
            A_total = fit_cd.iloc[0] - fit_cd.iloc[-1]
            C = fit_cd.iloc[-1]
            k0_fast = estimate_initial_k(fit_time, fit_cd)
            k0_slow = k0_fast / 10
            initial_guess = [0.5 * A_total, k0_fast, 0.5 * A_total, k0_slow, C]
            params, pcov = curve_fit(double_exponential, fit_time, fit_cd, p0=initial_guess, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            t_half1 = np.log(2) / params[1]
            t_half2 = np.log(2) / params[3]
            ratio_a1 = params[0] / (params[0] + params[2])
            fit_summary = (
                f"Fit params: A1={params[0]:.3f}±{perr[0]:.3f}, k1={params[1]:.3f}±{perr[1]:.3f} (t_half={t_half1:.3f}s), "
                f"A2={params[2]:.3f}±{perr[2]:.3f}, k2={params[3]:.3f}±{perr[3]:.3f} (t_half={t_half2:.3f}s), "
                f"c={params[4]:.3f}±{perr[4]:.3f}. A1 ratio={ratio_a1:.3f}")
            fit_func = double_exponential
            # For double exponential, let's report the dominant t_half as the "overall" t_half
            t_half = t_half1 if params[0] > params[2] else t_half2

        else:
            raise ValueError(f"Unsupported fit type: {fit_type}")

    except (RuntimeError, ValueError) as e:
        print(f"❌ Fit failed for type '{fit_type}': {e}")
        return None, None, None, "Fit failed.", None, None

    params_std_dev = np.sqrt(np.diag(pcov)) if pcov is not None else None
    return params, params_std_dev, fit_func, fit_summary, t_half, ratio_a1


def plot_combined_with_residuals(all_data, dead_times, plot_title, output_plot):
    """
    Generates a two-panel plot with raw data and fits in the top panel
    and residuals in the bottom panel.
    """
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    color_map = plt.get_cmap('Paired')

    # The loop has been simplified to iterate through a single list of dictionaries
    for i, data_dict in enumerate(all_data):
        # Unpack the dictionary for cleaner code
        filepath = data_dict['filepath']
        df_data = data_dict['df']
        params = data_dict['params']
        fit_type = data_dict['fit_type']
        label = data_dict['label']

        # Use a unique color for each dataset
        color = color_map(i * 2)
        darker_color = to_rgb(color_map(i * 2 + 1))  # Darker shade for the fit line

        # Use the filename basename to get the dead time from the dictionary
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        time = df_data['Time'] + dead_times.get(filename_base, 0)
        cd_signal = df_data['CircularDichroism']

        # Plot raw data
        ax_main.plot(time, cd_signal, '.', color=color, alpha=0.6, label=f"{label} Raw")

        # Plot fit and residuals
        fit_time = time
        fit_func_dict = {
            'exponential': exponential,
            'single_exponential_with_drift': single_exponential_with_drift,
            'double_exponential': double_exponential
        }
        if params is not None and fit_type in fit_func_dict:
            fit_func = fit_func_dict[fit_type]
            fit_vals = fit_func(fit_time, *params)
            ax_main.plot(fit_time, fit_vals, '-', color=darker_color, linewidth=2, label=f"{label} Fit")

            # Calculate and plot residuals
            residuals = cd_signal - fit_vals
            ax_res.plot(fit_time, residuals, 'o', color=darker_color, alpha=0.5, markersize=3)
        else:
            print(f"⚠️ No valid fit for {label}, skipping fit and residual plot.")

    # Axis labels & formatting
    ax_main.set_title(plot_title)
    ax_main.set_ylabel("Ellipticity (mdeg)")
    ax_main.legend(fontsize=8)
    ax_main.grid(True, linestyle='--', alpha=0.6)

    ax_res.set_ylabel("Residuals")
    ax_res.set_xlabel("Time (s)")
    ax_res.grid(True, linestyle='--', alpha=0.6)

    # Set y-axis limits for residuals to be symmetric around zero
    y_res_max = max(abs(ax_res.get_ylim()[0]), abs(ax_res.get_ylim()[1]))
    ax_res.set_ylim(-y_res_max, y_res_max)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()
    plt.close()


def read_dead_times(file_path):
    """Reads dead times from a specified file."""
    dead_times = {}
    if not os.path.exists(file_path):
        print(f"⚠️ Dead time file '{file_path}' not found. Using default dead time for all files.")
        return dead_times

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, time_str = parts
                try:
                    # Remove any 's' character and convert to integer
                    time_value = int(time_str.rstrip('s'))
                    dead_times[name] = time_value
                except ValueError:
                    print(f"⚠️ Invalid dead time format for file '{name}': '{time_str}'")
    return dead_times


# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/js/63/b11/kinetyka_2000s"

    # You can change this to 'exponential' or 'single_exponential_with_drift'
    fit_type = "double_exponential"
    default_dead_time = 0
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)

    results = []
    # This list will store dictionaries with all necessary data for each file
    all_data_for_plotting = []

    # Iterate through all CSV files in the specified folder
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.csv"))):
        try:
            df, wavelength_label = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)

            label = f"{filename_base}"

            # Perform the fit on the raw data
            params, perr, fit_func, fit_summary, t_half, ratio_a1 = fit_data(
                df['Time'], df['CircularDichroism'], fit_type, file_dead_time
            )

            # Store all data in a single dictionary and append it to our list
            all_data_for_plotting.append({
                'filepath': filepath,
                'df': df,
                'params': params,
                'fit_type': fit_type,
                'label': label
            })

            results.append(f"{os.path.basename(filepath)}: {fit_summary}")

        except Exception as e:
            error_msg = f"{os.path.basename(filepath)}: Failed to process - {str(e)}"
            print(error_msg)
            results.append(error_msg)

    # Save fitting results to a file
    fitting_results_path = os.path.join(folder_path, "fitting_results.txt")
    with open(fitting_results_path, "w") as f:
        f.write("\n".join(results))

    # After the loop, generate the final plot using the collected data
    if all_data_for_plotting:
        plot_combined_with_residuals(
            all_data=all_data_for_plotting,
            dead_times=dead_times_dict,
            plot_title=f"Kinetic fits ({wavelength_label})",
            output_plot=os.path.join(folder_path, "combined_fit_with_residuals.png")
        )
