import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re
import os
import glob
from io import StringIO


def read_data(file_content_or_path):
    """
    Reads data from a CSV file, handling 'ProDataCSV' and standard CSV formats.
    It extracts the wavelength label, processes time wraps, and cleans the data.
    """
    is_file_path = os.path.exists(file_content_or_path) if isinstance(file_content_or_path, str) else False

    if is_file_path:
        with open(file_content_or_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = file_content_or_path.splitlines()

    wavelength_label = "Wavelength (unknown nm)"

    if lines[0].strip() == "ProDataCSV":
        print(f"🧠 ProDataCSV structure detected.")

        # Extract wavelength
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

        # Find CircularDichroism section
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("❌ Could not locate 'CircularDichroism' section in the file.")

        header_line_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip().startswith("Time"):
                header_line_idx = j
                break
        else:
            raise ValueError("❌ Could not find data header (e.g., 'Time,Wavelength').")

        clean_data_lines = []
        numeric_line_pattern = re.compile(r'^-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?,-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?$')

        for k in range(header_line_idx + 1, len(lines)):
            stripped_line = lines[k].strip()
            if stripped_line == "": continue
            if numeric_line_pattern.fullmatch(stripped_line):
                clean_data_lines.append(stripped_line)

        if not clean_data_lines:
            raise ValueError("❌ No valid data rows found after header.")

        data_str = '\n'.join(clean_data_lines)
        df_full = pd.read_csv(StringIO(data_str), sep=",", header=None)
        df_full = df_full.dropna(how='all', axis=1)

        if df_full.shape[1] < 2:
            raise ValueError("Dataframe does not have enough columns.")

        df_full.rename(columns={0: 'Time', 1: 'CircularDichroism'}, inplace=True)

        if df_full.shape[1] > 2:
            df_full = df_full.iloc[:, :2]

        df_full = df_full.dropna(how='all')
        df_full['Time'] = pd.to_numeric(df_full['Time'], errors="coerce")
        df_full = df_full.dropna(subset=['Time'])

        time_col = df_full['Time'].values
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

        df = df.drop_duplicates(subset=['Time'], keep='first')

        # --- CHANGE 1: Remove Zeroes ---
        # Filters out rows where the signal is exactly 0 (spectrometer artifact)
        initial_len = len(df)
        df = df[df['CircularDichroism'] != 0]
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} zero-value data points.")
        # -------------------------------

        return df, wavelength_label

    else:
        print(f"📄 Standard CSV structure detected.")
        first_line = lines[0].strip()
        skiprows = 0
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            match = re.search(r'(\d+(?:\.\d+)?)', first_line)
            if match:
                wavelength_label = f"Wavelength: {match.group(1)} nm"
            else:
                wavelength_label = first_line
            skiprows = 1

        df_full = pd.read_csv(StringIO('\n'.join(lines)), skiprows=skiprows, sep=",")
        df_full = df_full.dropna(how='all', axis=1)
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

        # --- CHANGE 1: Remove Zeroes (for standard CSV too) ---
        # Assuming 2nd column is data
        data_col_name = df.columns[1]
        initial_len = len(df)
        df = df[df[data_col_name] != 0]
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} zero-value data points.")
        # ------------------------------------------------------

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

    A0 = max(intensity_series)
    C = min(intensity_series)
    half_max = (A0 + C) / 2
    half_max_index = np.abs(intensity_series - half_max).argmin()
    t_half = time_series.iloc[half_max_index]
    k_init = 1 / t_half if t_half > 0 else 1
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
    m0 = 0.0
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
    k0 = estimate_initial_k(time, intensity)
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
    time = pd.to_numeric(df['Time'], errors="coerce") + dead_time
    cd_signal = df['CircularDichroism']

    plt.figure(figsize=(9, 6))

    # Clean color palette
    c_raw = "#A0AAB2"
    c_smooth = "#1F77B4"
    c_fit = "#D62728"

    plt.plot(time, cd_signal, label='Raw CD Data', color=c_raw, alpha=0.4, linewidth=1)

    smoothed_time = None
    smoothed_signal = None
    current_t_half = None

    if smooth_method == 'moving_average':
        smoothed_signal = moving_average(cd_signal, window_size)
        smoothed_time = time[window_size - 1:]
        plt.plot(smoothed_time, smoothed_signal, label=f'Moving Avg (w={window_size})', color=c_smooth, linewidth=1.5)
    elif smooth_method == 'savitzky_golay':
        if window_size % 2 == 0:
            window_size += 1
        smoothed_signal = savgol_filter(cd_signal, window_size, polyorder)
        smoothed_time = time
        plt.plot(smoothed_time, smoothed_signal, label='Savitzky-Golay', color=c_smooth, linewidth=1.5)

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
            t_half = np.log(2) / params[1] if params[1] != 0 else float('inf')
            fitted_values = exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exponential Fit', color=c_fit, linestyle='--', linewidth=2)
            fit_result = (f"Fit: t_half={t_half:.2f}s, k={params[1]:.5f}")
            current_t_half = t_half

    elif fit_type == 'exponential_drift':
        params, errors = fit_exponential_with_drift(fit_time, fit_cd)
        if params is not None:
            t_half = np.log(2) / params[1] if params[1] != 0 else float('inf')
            fitted_values = single_exponential_with_drift(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Exp + Drift Fit', color="#FF7F0E", linestyle='--', linewidth=2)
            fit_result = (f"Fit: t_half={t_half:.2f}s, k={params[1]:.5f}")
            current_t_half = t_half

    elif fit_type == 'double_exponential':
        params, errors = fit_double_exponential(fit_time, fit_cd)
        if params is not None:
            t_half_k1 = np.log(2) / params[1] if params[1] != 0 else float('inf')
            t_half_k2 = np.log(2) / params[3] if params[3] != 0 else float('inf')
            fitted_values = double_exponential(fit_time, *params)
            plt.plot(fit_time, fitted_values, label='Double Exp Fit', color="#2CA02C", linestyle='--', linewidth=2)
            fit_result = (f"Fit: t_half1={t_half_k1:.2f}s, t_half2={t_half_k2:.2f}s")
            current_t_half = (t_half_k1, t_half_k2)

    elif fit_type == 'linear':
        slope, intercept = fit_linear(fit_time, fit_cd)
        fitted_values = slope * fit_time + intercept
        plt.plot(fit_time, fitted_values, label='Linear Fit', color="#9467BD", linestyle='--', linewidth=2)
        fit_result = f"Linear fit: slope={slope:.7f}"

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Ellipticity (mdeg)', fontsize=12)
    plt.title(f'CD Kinetics ({wavelength_label})', fontsize=14, fontweight='semibold')

    # Legend anchored neatly outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='#F8F9FA')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    return fit_result, smoothed_time, smoothed_signal, fit_time, fitted_values, current_t_half


def read_dead_times(file_path):
    dead_times = {}
    if not os.path.exists(file_path):
        print(f"⚠️ Dead time file '{file_path}' not found. Using default.")
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
                    pass
    return dead_times


def plot_all_folder_files(combined_curves, output_path):
    """
    Plots smoothed curves from all files in the directory onto a single summary plot.
    Uses high-contrast, visually distinct colors and places the legend outside the plot area.
    """
    if not combined_curves:
        print("⚠️ No datasets found to generate summary plot.")
        return

    plt.figure(figsize=(13, 8))
    num_traces = len(combined_curves)

    # Choose color maps dynamically based on total files
    if num_traces <= 10:
        colors = [plt.get_cmap('tab10')(i) for i in range(num_traces)]
    elif num_traces <= 20:
        colors = [plt.get_cmap('tab20')(i) for i in range(num_traces)]
    else:
        colors = [plt.get_cmap('turbo')(i) for i in np.linspace(0.05, 0.95, num_traces)]

    for idx, (smoothed_time, smoothed_signal, label) in enumerate(combined_curves):
        plt.plot(smoothed_time, smoothed_signal, label=label, color=colors[idx], linewidth=1.8)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Ellipticity (mdeg)', fontsize=14)
    plt.title('All Processed Kinetics (Overview)', fontsize=16, fontweight='semibold')

    # Clear, outside-anchored legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=True, title="Datasets")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', labelsize=12)
    plt.margins(0.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"✅ Summary plot of all folder files saved to: {output_path}")
    plt.close()


# BATCH PROCESSING SCRIPT
if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/CD/comparison/CD"

    smooth_method = "savitzky_golay"
    window_size = 15
    polyorder = 3
    default_dead_time = 25
    dead_time_file = os.path.join(folder_path, "dead_times.txt")
    dead_times_dict = read_dead_times(dead_time_file)

    fit_type = None
    fit_start = 0
    fit_end = 2000

    results = []
    combined_curves = []
    fitted_curves = []
    raw_curves = []
    all_t_half_values = []

    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df, wavelength_label_from_file = read_data(filepath)
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            file_dead_time = dead_times_dict.get(filename_base, default_dead_time)
            label = f"{filename_base}_{file_dead_time}"

            time_with_dead_time = df['Time']
            cd_signal = df['CircularDichroism']

            raw_curves.append((time_with_dead_time, cd_signal, str(label)))

            out_path = filepath[:-4] + "_fit.png"
            fit_summary, smoothed_time, smoothed_signal, fit_time, fit_vals, current_t_half = plot_data(
                df,
                wavelength_label=wavelength_label_from_file,
                smooth_method=smooth_method,
                window_size=window_size,
                polyorder=polyorder,
                output_plot=out_path,
                dead_time=file_dead_time,
                fit_type=fit_type,
                fit_start=fit_start,
                fit_end=fit_end
            )

            # Store smoothed curves for folder-wide plotting
            if smoothed_time is not None and smoothed_signal is not None:
                combined_curves.append((smoothed_time, smoothed_signal, str(label)))
            else:
                # Fallback to raw data if smoothing wasn't run
                combined_curves.append((time_with_dead_time + file_dead_time, cd_signal, str(label)))

            if fit_time is not None and fit_vals is not None:
                fitted_curves.append((fit_time, fit_vals, str(label)))

            results.append(f"{os.path.basename(filepath)}: {fit_summary}")

            # Collect T-half stats
            if fit_type == 'double_exponential' and current_t_half:
                t1, t2 = current_t_half
                all_t_half_values.extend([t1, t2])
            elif current_t_half is not None and np.isfinite(current_t_half):
                all_t_half_values.append(current_t_half)

        except Exception as e:
            print(f"{os.path.basename(filepath)}: Failed - {str(e)}")

    # Save Stats
    with open(os.path.join(folder_path, "fitting_results.txt"), "w") as f:
        f.write("\n".join(results))
        if all_t_half_values:
            mean_t = np.mean(all_t_half_values)
            std_t = np.std(all_t_half_values)
            f.write(f"\n\nMean t_half: {mean_t:.2f}s ± {std_t:.2f}s")

    # --- Plot ALL folder files (Smoothed Data Summary) ---
    all_files_plot_path = os.path.join(folder_path, "combined_processed_kinetics.png")
    plot_all_folder_files(combined_curves, all_files_plot_path)
