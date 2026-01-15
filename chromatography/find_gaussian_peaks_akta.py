import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION SECTION
# ==========================================
FILENAME = '/home/matifortunka/Documents/JS/Aria/Tm1570.asc'

# 1. Truncate Data (X-Axis/Volume)
# The baseline and fit will ONLY be calculated within this range.
# Set to (start_ml, end_ml) or None.
TRUNCATE_RANGE = (40, 130)

# 2. Baseline Correction
# 'linear'   : draws a straight line between the start and end of the data.
# 'min'      : finds the minimum value in the range and subtracts it.
# 'constant' : subtracts the specific number defined in BASELINE_CONSTANT below.
# 'none'     : no subtraction (baseline is 0).
BASELINE_MODE = 'constant'

# Only used if BASELINE_MODE = 'constant'
# This value will be subtracted from the signal.
BASELINE_CONSTANT = 12.5

# 3. Manual Peak Centers
# List the rough Volume (ml) positions where you expect peaks.
# Example: [42.5, 46.1]
# Set to None if you want the program to find peaks automatically.
MANUAL_PEAK_CENTERS = [65, 85, 95]
# MANUAL_PEAK_CENTERS = None

# 4. Automatic Detection Settings (Used only if MANUAL_PEAK_CENTERS is None)
MIN_PEAK_HEIGHT = 25.0  # mAU above baseline


# ==========================================
# MAIN PROGRAM
# ==========================================

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        mu = params[i + 1]
        sigma = params[i + 2]
        y += gaussian(x, amp, mu, sigma)
    return y


def load_and_process_data(filename):
    print(f"Loading {filename}...")
    try:
        # Read file, skipping header info safely
        df = pd.read_csv(filename, sep='\t', skiprows=4, header=None, on_bad_lines='skip')

        # Work on a clean copy of the first two columns
        df_work = df.iloc[:, [0, 1]].copy()
        df_work.columns = ['vol', 'signal']

        # Clean and convert data
        for col in ['vol', 'signal']:
            df_work[col] = df_work[col].astype(str).str.strip().str.replace(',', '.')
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        # Remove empty or bad rows
        df_clean = df_work.dropna()

        x = df_clean['vol'].values
        y = df_clean['signal'].values

        if len(x) == 0:
            print("Error: No valid data found in file.")
            return None, None

        return x, y

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


def main():
    x_raw, y_raw = load_and_process_data(FILENAME)
    if x_raw is None: return

    # --- 1. Truncate Data ---
    if TRUNCATE_RANGE:
        mask = (x_raw >= TRUNCATE_RANGE[0]) & (x_raw <= TRUNCATE_RANGE[1])
        x_data = x_raw[mask]
        y_data = y_raw[mask]
        print(f"Data truncated to {TRUNCATE_RANGE} ml. ({len(x_data)} points)")
        if len(x_data) < 5:
            print("Error: Not enough data points in the selected range.")
            return
    else:
        x_data, y_data = x_raw, y_raw

    # --- 2. Baseline Calculation ---
    baseline = np.zeros_like(y_data)

    if BASELINE_MODE == 'min':
        val = np.min(y_data)
        baseline = np.full_like(y_data, val)
        print(f"Baseline Mode: Min (Value={val:.3f})")

    elif BASELINE_MODE == 'constant':
        baseline = np.full_like(y_data, BASELINE_CONSTANT)
        print(f"Baseline Mode: Constant (Value={BASELINE_CONSTANT})")

    elif BASELINE_MODE == 'linear':
        # Fit a line between the average of the first 5 and last 5 points
        n_avg = 5
        start_y = np.mean(y_data[:n_avg])
        end_y = np.mean(y_data[-n_avg:])
        start_x = np.mean(x_data[:n_avg])
        end_x = np.mean(x_data[-n_avg:])

        slope = (end_y - start_y) / (end_x - start_x)
        c = start_y - slope * start_x
        baseline = slope * x_data + c
        print("Baseline Mode: Linear")

    y_corrected = y_data - baseline

    # --- 3. Peak Initialization ---
    initial_guesses = []

    if MANUAL_PEAK_CENTERS is not None and len(MANUAL_PEAK_CENTERS) > 0:
        print(f"Using MANUAL peak centers: {MANUAL_PEAK_CENTERS}")

        for center_vol in MANUAL_PEAK_CENTERS:
            # 1. Find the index in x_data closest to the user's manual volume
            # We clip the index to ensure we don't crash if the manual peak is slightly out of bounds
            idx = (np.abs(x_data - center_vol)).argmin()

            # 2. Estimate Amplitude from the actual data height at that point
            est_amp = y_corrected[idx]

            # 3. Initial Sigma guess (0.5 is a standard starting width)
            est_sigma = 0.5

            initial_guesses.extend([est_amp, center_vol, est_sigma])
    else:
        # Automatic Detection
        print(f"Using AUTOMATIC peak detection (Threshold: {MIN_PEAK_HEIGHT} mAU)")
        peaks_indices, properties = find_peaks(y_corrected, height=MIN_PEAK_HEIGHT, width=1)

        for i in range(len(peaks_indices)):
            height = properties['peak_heights'][i]
            pos = x_data[peaks_indices[i]]
            initial_guesses.extend([height, pos, 0.5])

    if not initial_guesses:
        print("No peaks found. Check TRUNCATE_RANGE or lower MIN_PEAK_HEIGHT.")
        initial_guesses = [np.max(y_corrected), np.mean(x_data), 1.0]

    # --- 4. Gaussian Fitting ---
    print(f"Fitting {len(initial_guesses) // 3} Gaussian(s)...")
    try:
        # Constrain parameters: Amplitude > 0, Sigma > 0. Mean is unconstrained.
        num_peaks = len(initial_guesses) // 3
        lower_bounds = [0, -np.inf, 0] * num_peaks
        upper_bounds = [np.inf, np.inf, np.inf] * num_peaks

        popt, pcov = curve_fit(multi_gaussian, x_data, y_corrected, p0=initial_guesses,
                               bounds=(lower_bounds, upper_bounds), maxfev=50000)
    except Exception as e:
        print(f"Fitting failed: {e}")
        popt = initial_guesses

    # --- 5. Visualization ---
    plt.figure(figsize=(10, 6))

    # Raw Data
    plt.plot(x_data, y_data, 'k.', markersize=4, label='Raw Data', alpha=0.3)

    # Baseline
    plt.plot(x_data, baseline, 'g--', label=f'Baseline ({BASELINE_MODE})', alpha=0.6)

    # Total Fit (Sum of Gaussians + Baseline)
    y_fit_sum = multi_gaussian(x_data, *popt) + baseline
    plt.plot(x_data, y_fit_sum, 'r-', linewidth=2, label='Total Fit')

    # Individual Peaks
    print("\n--- Peak Results ---")
    for i in range(num_peaks):
        amp = popt[3 * i]
        mu = popt[3 * i + 1]
        sigma = popt[3 * i + 2]

        print(f"Peak {i + 1}: Center = {mu:.3f} ml | Height = {amp:.3f} mAU | Width (sigma) = {sigma:.3f}")

        # Plot individual component
        y_comp = gaussian(x_data, amp, mu, sigma) + baseline
        plt.plot(x_data, y_comp, '--', alpha=0.8, label=f'Peak @ {mu:.1f}ml')
        plt.fill_between(x_data, baseline, y_comp, alpha=0.2)

    plt.xlabel('Volume (ml)')
    plt.ylabel('Absorbance (mAU)')
    plt.title(f'Gaussian Fit: {FILENAME}\n(Range: {TRUNCATE_RANGE} ml)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FILENAME[:-4]+".png")
    plt.show()


if __name__ == "__main__":
    main()