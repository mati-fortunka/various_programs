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
# The fit and purity check will ONLY happen inside this range.
TRUNCATE_RANGE = (40, 130)

# 2. Baseline Correction
# 'linear' (slope), 'min' (flat), 'constant' (manual value), or 'none'
BASELINE_MODE = 'constant'
BASELINE_CONSTANT = 12.5

# 3. Manual Peak Centers
# Set to None for automatic detection.
# Example: [42.5, 44.2, 46.0]
MANUAL_PEAK_CENTERS = [65, 85, 95]

# 4. Purity / Overlap Analysis
# The maximum allowed contribution from other peaks (0.05 = 5%).
PURITY_THRESHOLD = 0.1
# How many of the biggest peaks to analyze for purity?
MAX_PEAKS_TO_ANALYZE = 4

# 5. Automatic Detection (used if Manual Centers is None)
MIN_PEAK_HEIGHT = 25.0  # mAU above baseline


# ==========================================
# CORE FUNCTIONS
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
        # Skip first 4 rows of header, read tab-separated
        df = pd.read_csv(filename, sep='\t', skiprows=4, header=None, on_bad_lines='skip')

        # Take first 2 columns safely
        df_work = df.iloc[:, [0, 1]].copy()
        df_work.columns = ['vol', 'signal']

        # Clean: remove spaces, replace comma with dot, convert to float
        for col in ['vol', 'signal']:
            df_work[col] = df_work[col].astype(str).str.strip().str.replace(',', '.')
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        # Drop empty/bad rows
        df_clean = df_work.dropna()

        return df_clean['vol'].values, df_clean['signal'].values
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


def calculate_baseline(x, y, mode, constant_val):
    if mode == 'min':
        return np.full_like(y, np.min(y))
    elif mode == 'constant':
        return np.full_like(y, constant_val)
    elif mode == 'linear':
        n = 5  # average n points at edges
        if len(x) > n * 2:
            start_y = np.mean(y[:n])
            end_y = np.mean(y[-n:])
            start_x = np.mean(x[:n])
            end_x = np.mean(x[-n:])
            slope = (end_y - start_y) / (end_x - start_x)
            c = start_y - slope * start_x
            return slope * x + c
        return np.full_like(y, np.min(y))
    return np.zeros_like(y)


# ==========================================
# MAIN
# ==========================================
def main():
    x_raw, y_raw = load_and_process_data(FILENAME)
    if x_raw is None: return

    # --- 1. Truncate Data ---
    if TRUNCATE_RANGE:
        mask = (x_raw >= TRUNCATE_RANGE[0]) & (x_raw <= TRUNCATE_RANGE[1])
        x_data = x_raw[mask]
        y_data = y_raw[mask]
        print(f"Data range: {TRUNCATE_RANGE} ml ({len(x_data)} points)")
        if len(x_data) < 10:
            print("Error: Range too narrow or empty.")
            return
    else:
        x_data, y_data = x_raw, y_raw

    # --- 2. Baseline ---
    baseline = calculate_baseline(x_data, y_data, BASELINE_MODE, BASELINE_CONSTANT)
    y_corrected = y_data - baseline

    # --- 3. Peak Initialization ---
    initial_guesses = []

    if MANUAL_PEAK_CENTERS:
        print(f"Using Manual Centers: {MANUAL_PEAK_CENTERS}")
        for center in MANUAL_PEAK_CENTERS:
            # Find closest real data point to get amplitude estimate
            idx = (np.abs(x_data - center)).argmin()
            initial_guesses.extend([y_corrected[idx], center, 0.5])
    else:
        print("Using Automatic Detection...")
        peaks, props = find_peaks(y_corrected, height=MIN_PEAK_HEIGHT)
        for i in range(len(peaks)):
            initial_guesses.extend([props['peak_heights'][i], x_data[peaks[i]], 0.5])

    if not initial_guesses:
        print("No peaks found. Using fallback guess.")
        initial_guesses = [np.max(y_corrected), np.mean(x_data), 1.0]

    # --- 4. Fitting ---
    num_peaks = len(initial_guesses) // 3
    print(f"Fitting {num_peaks} Gaussian(s)...")

    try:
        # Bounds: Amp>0, Mean=any, Sigma>0
        lower = [0, -np.inf, 0] * num_peaks
        upper = [np.inf, np.inf, np.inf] * num_peaks
        popt, _ = curve_fit(multi_gaussian, x_data, y_corrected, p0=initial_guesses,
                            bounds=(lower, upper), maxfev=50000)
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = initial_guesses

    # --- 5. Purity Analysis ---
    # Reconstruct individual Gaussians
    gaussians = []
    for i in range(num_peaks):
        g = {
            'amp': popt[3 * i],
            'mu': popt[3 * i + 1],
            'sigma': popt[3 * i + 2],
            'id': i + 1
        }
        gaussians.append(g)

    # Sort by Amplitude (Height) descending, keep top N
    sorted_gaussians = sorted(gaussians, key=lambda k: k['amp'], reverse=True)
    top_peaks = sorted_gaussians[:MAX_PEAKS_TO_ANALYZE]

    # Calculate Total Fit Curve
    y_fit_total = multi_gaussian(x_data, *popt)

    print(f"\n--- Purity Analysis (Threshold: {PURITY_THRESHOLD * 100}%) ---")
    purity_ranges = []  # Store for plotting (vol_start, vol_end, peak_id)

    for p in top_peaks:
        # 1. Generate this peak's curve
        y_this = gaussian(x_data, p['amp'], p['mu'], p['sigma'])

        # 2. Sum of ALL OTHER peaks
        y_others = y_fit_total - y_this

        # 3. Calculate Contamination Ratio: Others / Total
        # Add small epsilon to avoid divide by zero
        ratio = y_others / (y_fit_total + 1e-9)

        # 4. Find center index
        center_idx = (np.abs(x_data - p['mu'])).argmin()

        # 5. Check purity at center
        if ratio[center_idx] > PURITY_THRESHOLD:
            print(
                f"Peak {p['id']} (Vol {p['mu']:.2f}): HEAVILY OVERLAPPED. Center impurity {ratio[center_idx] * 100:.1f}%.")
            continue

        # 6. Scan Left
        left_idx = center_idx
        while left_idx > 0:
            if ratio[left_idx] > PURITY_THRESHOLD:
                break
            left_idx -= 1

        # 7. Scan Right
        right_idx = center_idx
        while right_idx < len(x_data) - 1:
            if ratio[right_idx] > PURITY_THRESHOLD:
                break
            right_idx += 1

        v_start = x_data[left_idx]
        v_end = x_data[right_idx]

        print(f"Peak {p['id']} (Vol {p['mu']:.2f}): Pure Range [{v_start:.2f} -> {v_end:.2f}] ml")
        purity_ranges.append((v_start, v_end, p['id']))

    # --- 6. Plotting ---
    plt.figure(figsize=(12, 7))

    # Plot Data & Baseline
    plt.plot(x_data, y_data, 'k.', markersize=3, alpha=0.3, label='Raw Data')
    plt.plot(x_data, baseline, 'g--', linewidth=1, label='Baseline')
    plt.plot(x_data, y_fit_total + baseline, 'r-', linewidth=2, alpha=0.9, label='Total Fit')

    # Plot Components
    for i, p in enumerate(gaussians):
        y_comp = gaussian(x_data, p['amp'], p['mu'], p['sigma']) + baseline
        plt.plot(x_data, y_comp, '--', linewidth=1, alpha=0.7, label=f'Peak {p["id"]}')

    # Plot Pure Ranges (Shaded Areas)
    for (start, end, pid) in purity_ranges:
        # Find mask for this range
        mask_range = (x_data >= start) & (x_data <= end)
        # Fill between baseline and Total Fit (or individual peak)
        # Using Total Fit helps visualize the "collectible" area
        plt.fill_between(x_data, baseline, y_fit_total + baseline,
                         where=mask_range, color='green', alpha=0.2)
        # Add text label
        mid_x = (start + end) / 2
        # Y position: roughly at 20% of peak height
        mid_y = baseline[mask_range][0] + (np.max(y_fit_total) / 5)
        plt.text(mid_x, mid_y, f"Pure\nP{pid}", ha='center', fontsize=8, color='green')

    plt.xlabel('Volume (ml)')
    plt.ylabel('Absorbance (mAU)')
    plt.title(f'Gaussian Fit & Purity Analysis (<{PURITY_THRESHOLD * 100}% overlap)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FILENAME[:-4]+f"_{PURITY_THRESHOLD}.png")
    plt.show()


if __name__ == "__main__":
    main()