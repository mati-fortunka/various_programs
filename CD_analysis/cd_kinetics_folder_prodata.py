import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob  # Re-import glob for folder processing
from io import StringIO  # Keep StringIO for flexibility in read_data, though not used in main for file paths


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
    # Determine if input is file path or content string
    is_file_path = os.path.exists(file_content_or_path) if isinstance(file_content_or_path, str) else False

    if is_file_path:
        with open(file_content_or_path, 'r') as file:
            lines = file.readlines()
    else:  # Assume it's file content string (e.g., for direct string input or testing)
        lines = file_content_or_path.splitlines()

    wavelength_label = "Wavelength (unknown nm)"  # Default label

    if lines[0].strip() == "ProDataCSV":
        print(f"🧠 ProDataCSV structure detected.")

        # Extract wavelength from header (e.g., "Wavelength,Wavelength: 222nm,,Bandwidth: 1nm")
        wavelength_found = False
        for line in lines:
            if "Wavelength,Wavelength:" in line:
                match = re.search(r'Wavelength:\s*(\d+(?:\.\d+)?)\s*nm', line)
                if match:
                    wavelength_label = f"Wavelength: {match.group(1)} nm"
                    wavelength_found = True
                    break
        if not wavelength_found:
            print("⚠️ Wavelength not found in header. Using default label.")

        # Find where the CircularDichroism section starts
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("❌ Could not locate 'CircularDichroism' section in the file.")

        # Look for the header line like "Time,Wavelength"
        header_line_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip().startswith("Time"):  # More robust check for "Time,Wavelength"
                header_line_idx = j
                break
        else:
            raise ValueError("❌ Could not find data header (e.g., 'Time,Wavelength').")

        # Collect only lines that appear to be valid data rows (two numeric fields)
        clean_data_lines = []
        # Define a regex for a line containing two numeric values separated by a comma
        # This will filter out lines like ",222," or any other non-data lines.
        numeric_line_pattern = re.compile(r'^-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?,-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?$')

        # Start looking for data *after* the header line (Time,Wavelength)
        # We iterate through the lines and only add those that strictly match the numeric pattern.
        for k in range(header_line_idx + 1, len(lines)):
            stripped_line = lines[k].strip()
            if stripped_line == "":  # Skip empty lines
                continue

            if numeric_line_pattern.fullmatch(stripped_line):
                clean_data_lines.append(stripped_line)
            # If a line doesn't match and we've already started collecting data,
            # it indicates the end of the data block or a malformed line.
            # For robustness, we'll only collect lines that strictly match the pattern.

        if not clean_data_lines:
            raise ValueError("❌ No valid data rows found after header.")

        data_str = '\n'.join(clean_data_lines)
        # Read the CSV. Since we're starting from the actual data, there's no header to parse from this string.
        # We assume the first column is Time and the second is CD signal.
        df_full = pd.read_csv(StringIO(data_str), sep=",", header=None)

        # Drop any extra columns that might appear due to trailing commas in the original file
        df_full = df_full.dropna(how='all', axis=1)

        # Ensure we have at least two columns for Time and CircularDichroism
        if df_full.shape[1] < 2:
            raise ValueError("Dataframe does not have enough columns after parsing. Expected at least 2.")

        # Rename columns for clarity, assuming fixed positions
        df_full.rename(columns={0: 'Time', 1: 'CircularDichroism'}, inplace=True)

        # Drop any columns beyond the second one if they exist (e.g., from trailing commas in data lines)
        if df_full.shape[1] > 2:
            df_full = df_full.iloc[:, :2]

        # Drop empty rows that might have been introduced if there were blank lines in data_lines_to_parse
        df_full = df_full.dropna(how='all')

        # Convert the 'Time' column to numeric
        df_full['Time'] = pd.to_numeric(df_full['Time'], errors="coerce")
        # Drop rows where Time is NaN (if conversion failed)
        df_full = df_full.dropna(subset=['Time'])

        # Handle time wrap (if time resets or decreases)
        time_col = df_full['Time'].values
        wrap_index = None
        # Start checking from the second element, as time_col[0] is the first data point
        for i in range(1, len(time_col)):
            if time_col[i] <= time_col[i - 1]:
                wrap_index = i
                break

        if wrap_index is not None:
            print(f"Detected time wrap at row {wrap_index}. Truncating data.")
            df = df_full.iloc[:wrap_index].copy()
        else:
            df = df_full.copy()

        # Drop duplicates based on the 'Time' column
        df = df.drop_duplicates(subset=['Time'], keep='first')
        return df, wavelength_label

    else:
        print(f"📄 Standard CSV structure detected.")
        # Fallback to original logic for standard CSVs
        first_line = lines[0].strip()
        skiprows = 0
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            print(f"Header detected: '{first_line}'")
            match = re.search(r'(\d+(?:\.\d+)?)', first_line)
            if match:
                wavelength_label = f"Wavelength: {match.group(1)} nm"
            else:
                wavelength_label = first_line  # Keep original behavior if no wavelength found
            skiprows = 1
        else:
            print("⚠️ No header detected or malformed header. Assuming data starts immediately.")
            skiprows = 0

        df_full = pd.read_csv(StringIO('\n'.join(lines)), skiprows=skiprows, sep=",")
        df_full = df_full.dropna(how='all', axis=1)

        # Convert the first column (Time) to numeric before checking for wrap
        df_full.iloc[:, 0] = pd.to_numeric(df_full.iloc[:, 0], errors="coerce")

        time_col = df_full.iloc[:, 0].values
        wrap_index = None
        for i in range(1, len(time_col)):
            if time_col[i] <= time_col[i - 1]:
                wrap_index = i
                break

        if wrap_index is not None:
            print(f"Detected time wrap at row {wrap_index}. Truncating data.")
            df = df_full.iloc[:wrap_index].copy()
        else:
            df = df_full.copy()

        df = df.drop_duplicates(subset=df.columns[0], keep='first')
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
    # Ensure time and intensity are pandas Series for consistent indexing
    time_series = pd.Series(time)
    intensity_series = pd.Series(intensity)

    A0 = max(intensity_series)
    C = min(intensity_series)
    half_max = (A0 + C) / 2
    # Find the index where the intensity is closest to half_max
    half_max_index = np.abs(intensity_series - half_max).argmin()
    t_half = time_series.iloc[half_max_index]
    k_init = 1 / t_half if t_half > 0 else 1  # Avoid division by zero
    return k_init


def fit_exponential(time, intensity):
    """Fits data to a single exponential function."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    initial_guess = [A0, k0, C]
    try:
        popt, pcov = curve_fit(exponential, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential fit failed.")
        return None, None


def fit_exponential_with_drift(time, intensity):
    """Fits data to a single exponential function with linear drift."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    m0 = 0.0  # Initial guess for drift
    initial_guess = [A0, k0, C, m0]
    try:
        popt, pcov = curve_fit(single_exponential_with_drift, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential with drift fit failed.")
        return None, None


def fit_double_exponential(t, A1, k1, A2, k2, c):
    """Double exponential decay function."""
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def estimate_initial_k(time, intensity):
    """Estimates an initial rate constant (k) for exponential fitting."""
    # Ensure time and intensity are pandas Series for consistent indexing
    time_series = pd.Series(time)
    intensity_series = pd.Series(intensity)

    A0 = max(intensity_series)
    C = min(intensity_series)
    half_max = (A0 + C) / 2
    # Find the index where the intensity is closest to half_max
    half_max_index = np.abs(intensity_series - half_max).argmin()
    t_half = time_series.iloc[half_max_index]
    k_init = 1 / t_half if t_half > 0 else 1  # Avoid division by zero
    return k_init


def fit_exponential(time, intensity):
    """Fits data to a single exponential function."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    initial_guess = [A0, k0, C]
    try:
        popt, pcov = curve_fit(exponential, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential fit failed.")
        return None, None


def fit_exponential_with_drift(time, intensity):
    """Fits data to a single exponential function with linear drift."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)
    m0 = 0.0  # Initial guess for drift
    initial_guess = [A0, k0, C, m0]
    try:
        popt, pcov = curve_fit(single_exponential_with_drift, time, intensity, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Exponential with drift fit failed.")
        return None, None


def fit_double_exponential(time, intensity):
    """Fits data to a double exponential function."""
    A0 = max(intensity)
    C = min(intensity)
    k0 = estimate_initial_k(time, intensity)  # Use a single k0 for initial guess
    # Common heuristic: k2 is often slower than k1
    initial_guess = [0.7 * A0, k0, 0.3 * A0, k0 / 10 if k0 > 0 else 0.001, C]
    try:
        popt, pcov = curve_fit(double_exponential, time, intensity, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print("Double exponential fit failed.")
        return None, None


def fit_linear(time, intensity):
    """Fits data to a linear function."""
    coeffs = np.polyfit(time, intensity, 1)
    return coeffs


def plot_data(df, wavelength_label, smooth_method=None, window_size=5, polyorder=2,
              output_plot="out.png", dead_time=20,
              fit_type='exponential', fit_start=None, fit_end=None):
    """
    Plots the raw, smoothed, and fitted CD kinetics data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'Time' and 'CircularDichroism' data.
        wavelength_label (str): The label for the wavelength (e.g., "Wavelength: 222 nm").
        smooth_method (str, optional): Smoothing method ('moving_average' or 'savitzky_golay'). Defaults to None.
        window_size (int, optional): Window size for smoothing. Defaults to 5.
        polyorder (int, optional): Polynomial order for Savitzky-Golay. Defaults to 2.
        output_plot (str, optional): Filename to save the plot. Defaults to "out.png".
        dead_time (int, optional): Dead time to add to the time axis. Defaults to 20.
        fit_type (str, optional): Type of fit to apply ('exponential', 'exponential_drift', 'double_exponential', 'linear'). Defaults to 'exponential'.
        fit_start (float, optional): Start time for fitting (absolute value on adjusted time axis). Defaults to None (start of data).
        fit_end (float, optional): End time for fitting (absolute value on adjusted time axis). Defaults to None (end of data).

    Returns:
        tuple: A tuple containing:
            - str: Summary of the fit results.
            - pandas.Series: Smoothed time data (or None if no smoothing).
            - pandas.Series: Smoothed signal data (or None if no smoothing).
            - pandas.Series: Time data used for fitting (or None if no fit).
            - numpy.ndarray: Fitted signal values (or None if no fit).
            - float: t_half value from the fit, or None if not applicable/failed.
    """

    time = pd.to_numeric(df['Time'], errors="coerce") + dead_time  # Use named column
    cd_signal = df['CircularDichroism']  # Use named column

    plt.figure(figsize=(8, 5))
    plt.plot(time, cd_signal, label='Raw CD Data', color='gray', alpha=0.6)

    smoothed_time = None
    smoothed_signal = None
    current_t_half = None  # Initialize t_half for return

    if smooth_method == 'moving_average':
        smoothed_signal = moving_average(cd_signal, window_size)
        # Moving average reduces length by window_size - 1, adjust time accordingly
        smoothed_time = time[window_size - 1:]
        plt.plot(smoothed_time, smoothed_signal, label=f'Moving Avg (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        # Savitzky-Golay filter needs odd window size
        if window_size % 2 == 0:
            window_size += 1
            print(f"Adjusted Savitzky-Golay window_size to {window_size} (must be odd).")
        smoothed_signal = savgol_filter(cd_signal, window_size, polyorder)
        smoothed_time = time
        plt.plot(smoothed_time, smoothed_signal, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})',
                 color='red')

    # Define fit range based on absolute time values
    # If fit_start/fit_end are None, use the full range of the 'time' series
    actual_fit_start = fit_start if fit_start is not None else time.min()
    actual_fit_end = fit_end if fit_end is not None else time.max()

    mask = (time >= actual_fit_start) & (time <= actual_fit_end)
    fit_time = time[mask]
    fit_cd = cd_signal[mask]

    fit_result = ""
    fitted_values = None

    if fit_type == 'exponential':
        params, errors = fit_exponential(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1] if params[1] != 0 else float('inf')  # Handle k=0
            fitted_values = exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exponential Fit', color='green')
            fit_result = (
                f"Exponential fit: A={params[0]:.7f}±{errors[0]:.7f}, "
                f"k={params[1]:.7f}±{errors[1]:.7f}, c={params[2]:.7f}±{errors[2]:.7f}, "
                f"t_half={t_half:.2f}s"
            )
            current_t_half = t_half  # Store t_half

    elif fit_type == 'exponential_drift':
        params, errors = fit_exponential_with_drift(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1] if params[1] != 0 else float('inf')  # Handle k=0
            fitted_values = single_exponential_with_drift(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exp + Drift Fit', color='orange')
            fit_result = (
                f"Exp+Drift fit: A={params[0]:.7f}±{errors[0]:.7f}, "
                f"k={params[1]:.7f}±{errors[1]:.7f}, c={params[2]:.7f}±{errors[2]:.7f}, "
                f"m={params[3]:.7f}±{errors[3]:.7f}, t_half={t_half:.2f}s"
            )
            current_t_half = t_half  # Store t_half

    elif fit_type == 'double_exponential':
        params, errors = fit_double_exponential(fit_time, fit_cd)
        if params is not None:
            t_half_k1 = np.log(2) / params[1] if params[1] != 0 else float('inf')
            t_half_k2 = np.log(2) / params[3] if params[3] != 0 else float('inf')
            fitted_values = double_exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Double Exp Fit', color='brown')
            fit_result = (
                f"Double exp fit: A1={params[0]:.7f}±{errors[0]:.7f}, "
                f"k1={params[1]:.7f}±{errors[1]:.7f}, A2={params[2]:.7f}±{errors[2]:.7f}, "
                f"k2={params[3]:.7f}±{errors[3]:.7f}, c={params[4]:.7f}±{errors[4]:.7f}, "
                f"t_half_k1={t_half_k1:.2f}s, t_half_k2={t_half_k2:.2f}s"
            )
            current_t_half = (t_half_k1, t_half_k2)  # For double exponential, store the first t_half

    elif fit_type == 'linear':
        slope, intercept = fit_linear(fit_time, fit_cd)
        fitted_values = slope * fit_time + intercept
        plt.plot(fit_time, fitted_values, label='Linear Fit', color='purple')
        fit_result = f"Linear fit: slope={slope:.7f}, intercept={intercept:.7f}"
        # t_half is not applicable for linear fit, so current_t_half remains None

    plt.xlabel('Time (s)')
    plt.ylabel('Ellipticity (mdeg)')
    plt.title(f'CD Kinetics Over Time\n({wavelength_label})')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.close()

    return fit_result, smoothed_time, smoothed_signal, fit_time, fitted_values, current_t_half  # Return t_half


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
                    time_value = int(time_str.rstrip('s'))
                    dead_times[name] = time_value
                except ValueError:
                    print(f"⚠️ Invalid dead time format for file '{name}': '{time_str}'")
    return dead_times


# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    # IMPORTANT: Replace "/path/to/your/data/folder" with the actual path to your CSV files.
    folder_path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/8b1n/kinetics/cd/fuzja_october/2000s"
    # Example: If your files are in a subfolder named 'data' in the same directory as your script,
    # you might use: folder_path = os.path.join(os.path.dirname(__file__), 'data')

    smooth_method = 'savitzky_golay'
    window_size = 25
    polyorder = 3
    default_dead_time = 0  # Default dead time if not specified in dead_times.txt
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)  # Load dead times from file

    fit_type = "double_exponential"
    fit_start = 0
    fit_end = 2000

    results = []
    all_fit_params = []
    combined_curves = []
    fitted_curves = []
    raw_curves = []  # To store raw (time, signal, label)
    all_t_half_values = []  # New list to store t_half values

    # Iterate through all CSV files in the specified folder
    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            # Read data from the current file path
            df, wavelength_label_from_file = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            # Get dead time for the current file, or use default if not found
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)

            label = f"{filename_base}_{file_dead_time}"

            # Prepare data for plotting (time includes dead_time)
            time_with_dead_time = df['Time']  # Use named column
            cd_signal = df['CircularDichroism']  # Use named column

            raw_curves.append((time_with_dead_time, cd_signal, str(label)))

            # Define output path for the individual plot
            out_path = filepath[:-4] + "_fit.png"
            # Capture the new t_half return value
            fit_summary, smoothed_time, smoothed_signal, fit_time, fit_vals, current_t_half = plot_data(
                df,
                wavelength_label=wavelength_label_from_file,  # Pass the extracted label
                smooth_method=smooth_method,
                window_size=window_size,
                polyorder=polyorder,
                output_plot=out_path,
                dead_time=file_dead_time,
                fit_type=fit_type,
                fit_start=fit_start,
                fit_end=fit_end
            )

            if smoothed_time is not None and smoothed_signal is not None:
                combined_curves.append((smoothed_time, smoothed_signal, str(label)))
            if fit_time is not None and fit_vals is not None:
                fitted_curves.append((fit_time, fit_vals, str(label)))

            results.append(f"{os.path.basename(filepath)}: {fit_summary}")

            # Extract parameters from the fit summary string
            param_matches = re.findall(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)±", fit_summary)
            if param_matches:
                all_fit_params.append([float(val) for val in param_matches])

            # Add current_t_half to the list if it's not None
            if fit_type != 'double_exponential':
                if current_t_half is not None and np.isfinite(current_t_half):
                    all_t_half_values.append(current_t_half)

            else:
                t_1, t_2 = current_t_half
                if t_1 < t_2:
                    all_t_half_values.append(t_1)
                    all_t_half_values.append(t_2)
                else:
                    all_t_half_values.append(t_2)
                    all_t_half_values.append(t_1)

        except Exception as e:
            error_msg = f"{os.path.basename(filepath)}: Failed to process - {str(e)}"
            print(error_msg)
            results.append(error_msg)

    # Save fitting results to a file
    fitting_results_path = os.path.join(folder_path, "fitting_results.txt")
    with open(fitting_results_path, "w") as f:
        f.write("\n".join(results))

        if all_fit_params:
            param_array = np.array(all_fit_params)
            means = np.mean(param_array, axis=0)
            stds = np.std(param_array, axis=0)

            # Define labels based on fit_type for clarity
            if fit_type == 'exponential':
                labels = ['A', 'k', 'c']
            elif fit_type == 'exponential_drift':
                labels = ['A', 'k', 'c', 'm']
            elif fit_type == 'double_exponential':
                labels = ['A1', 'k1', 'A2', 'k2', 'c']
            elif fit_type == 'linear':
                labels = ['slope', 'intercept']
            else:
                labels = [f"param{i + 1}" for i in range(param_array.shape[1])]

            f.write("\n\nSummary of Fitted Parameters:\n")
            for param_label, mean, std in zip(labels, means, stds):
                f.write(f"{param_label}: mean={mean:.7f}, std={std:.7f}\n")

        # Add mean t_half to the results file
        if all_t_half_values:
            if fit_type == 'double_exponential':
                mean_t_half1 = np.mean(all_t_half_values[::2])
                std_t_half1 = np.std(all_t_half_values[::2])
                mean_t_half2 = np.mean(all_t_half_values[1::2])
                std_t_half2 = np.std(all_t_half_values[1::2])
                f.write(f"Mean t_half: {mean_t_half1:.2f}s ± {std_t_half1:.2f}s\nMean t_half: {mean_t_half2:.2f}s ± {std_t_half2:.2f}s")
            else:
                mean_t_half = np.mean(all_t_half_values)
                std_t_half = np.std(all_t_half_values)
                f.write(f"Mean t_half: {mean_t_half:.2f}s ± {std_t_half:.2f}s\n")
        else:
            f.write("\nNo t_half values available for averaging (e.g., no exponential fits performed).\n")

    # Plot combined raw + fitted curves
    if raw_curves and fitted_curves:
        plt.figure(figsize=(6, 5))

        color_map = plt.get_cmap('tab10')

        for idx, ((rtime, rsignal, label), (ftime, fsignal, _)) in enumerate(zip(raw_curves, fitted_curves)):
            color = color_map(idx % 10)
            plt.plot(rtime, rsignal, label=f"{label} (Raw)", alpha=0.5, color=color)
            plt.plot(ftime, fsignal, label=f"{label} (Fit)", linestyle='--', linewidth=1.5, color=color)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Ellipticity at 222 nm [mdeg]', fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # The legend is commented out in your original code for the combined plot,
        # but you can uncomment it if you want to see the labels for each curve.
        # plt.legend(fontsize=14, frameon=False)
        plt.tight_layout()

        # Save combined plot to a file in the specified folder
        png_path = os.path.join(folder_path, "combined_raw_fitted_plot.png")
        plt.savefig(png_path, dpi=600)
        plt.show()  # This command will display the plot in the output
        plt.close()
