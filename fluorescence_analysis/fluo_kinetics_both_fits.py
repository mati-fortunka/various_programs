import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re


# ===========================================================
# 1. Model Definitions
# ===========================================================

def exponential(t, A, k, c):
    """Single exponential decay model."""
    return A * np.exp(-k * t) + c


def double_exponential(t, A1, k1, A2, k2, c):
    """Double exponential decay model."""
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def sigmoid_model(t, y0, a, k, t_half):
    """Sigmoid growth model."""
    return y0 + a / (1 + np.exp(-k * (t - t_half)))


def exp_sigmoid_model(t, A_exp, k_exp, c_exp, y0_sig, a_sig, k_sig, t_half_sig):
    """Combines an exponential decay and a sigmoid growth model."""
    return exponential(t, A_exp, k_exp, c_exp) + sigmoid_model(t, y0_sig, a_sig, k_sig, t_half_sig)


def double_exp_sigmoid_model(t, A1_exp, k1_exp, A2_exp, k2_exp, c_exp, y0_sig, a_sig, k_sig, t_half_sig):
    """Combines a double exponential decay and a sigmoid growth model."""
    return double_exponential(t, A1_exp, k1_exp, A2_exp, k2_exp, c_exp) + sigmoid_model(t, y0_sig, a_sig, k_sig,
                                                                                        t_half_sig)


# ===========================================================
# 2. Data Processing
# ===========================================================

def read_data(filename):
    """Reads data from a CSV file, handling headers and potential data wraps."""
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            skiprows = 1
        else:
            skiprows = 0

    skiprows = 2

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
    df_full = df_full.dropna(how='all', axis=1)

    time_col = df_full.iloc[:, 0]
    initial_time = time_col.iloc[0]

    # Detect if time resets (data wrapping)
    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()
    if wrap_index > 1:
        print(f"Detected secondary block starting at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

    return df


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# ===========================================================
# 3. Helpers & Fitting
# ===========================================================

def estimate_initial_params(time, intensity, model_name):
    """Provides initial parameter guesses for different models."""
    # Ensure inputs are accessible via iloc/indexing
    if isinstance(intensity, np.ndarray):
        y_start, y_end = intensity[0], intensity[-1]
        y_min, y_max = intensity.min(), intensity.max()
    else:
        y_start, y_end = intensity.iloc[0], intensity.iloc[-1]
        y_min, y_max = intensity.min(), intensity.max()

    if model_name == 'exponential':
        A0 = y_start - y_end
        C = y_end
        target_y = C + A0 / 2
        idx = np.abs(intensity - target_y).argmin()
        t_val = time.iloc[idx] if hasattr(time, 'iloc') else time[idx]
        k0 = np.log(2) / t_val if t_val > 0 else 0.01
        return [A0, k0, C]

    elif model_name == 'sigmoid':
        y0_est = y_min
        a_est = y_max - y0_est
        target_y = y0_est + a_est / 2
        idx = np.abs(intensity - target_y).argmin()
        t_half_est = time.iloc[idx] if hasattr(time, 'iloc') else time[idx]
        k_est = 0.01
        return [y0_est, a_est, k_est, t_half_est]

    elif model_name == 'exp_sigmoid':
        return estimate_initial_params(time, intensity, 'exponential') + \
            estimate_initial_params(time, intensity, 'sigmoid')

    elif model_name == 'double_exp_sigmoid':
        A0 = y_start - y_end
        C = y_end
        double_exp_params = [0.7 * A0, 0.01, 0.3 * A0, 0.001, C]
        sig_params = estimate_initial_params(time, intensity, 'sigmoid')
        return double_exp_params + sig_params

    return None


def fit_model(model_func, time, intensity, initial_guess):
    try:
        popt, pcov = curve_fit(model_func, time, intensity, p0=initial_guess, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print(f"Fit failed for model: {model_func.__name__}")
        return None, None


# ===========================================================
# 4. Main Plotting Function
# ===========================================================

def plot_data(df, smooth_method=None, window_size=5, polyorder=2,
              output_plot="out.png", dead_time=0,
              fit_type='exponential',
              range1=(None, None),
              range2=(None, None)):
    # Prepare Data
    time = df.iloc[:, 0] + dead_time
    intensity = df.iloc[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(time, intensity, label='Raw Data', color='gray', alpha=0.4)

    # Smoothing
    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f'MovAvg (w={window_size})', color='blue', alpha=0.6)
        fit_source_time = time
        fit_source_int = intensity
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f'SavGol (w={window_size})', color='red', alpha=0.6)
        fit_source_time = time
        fit_source_int = intensity
    else:
        fit_source_time = time
        fit_source_int = intensity

    # -------------------------------------------------------
    # LOGIC: SPLIT REGIME FIT
    # -------------------------------------------------------
    if fit_type == 'split_regime':
        print(f"\n=== Running Split Regime Fit ===")

        # --- RANGE 1: Exponential vs Double Exponential ---
        if range1[0] is not None and range1[1] is not None:
            mask1 = (fit_source_time >= range1[0]) & (fit_source_time <= range1[1])
            t1 = fit_source_time[mask1]
            y1 = fit_source_int[mask1]

            if len(t1) > 0:
                print(f"\n--- Range 1 ({range1[0]}-{range1[1]}s): Exp vs Double Exp ---")

                # 1. Single Exponential
                guess_exp = estimate_initial_params(t1, y1, 'exponential')
                p_exp, err_exp = fit_model(exponential, t1, y1, guess_exp)
                if p_exp is not None:
                    plt.plot(t1, exponential(t1, *p_exp), color='lime', linewidth=2, label='Range 1: Exp Fit')
                    print(f"  [Exp Fit]")
                    print(f"     k     = {p_exp[1]:.5f}")
                    print(f"     t_1/2 = {np.log(2) / p_exp[1]:.5f} s")

                # 2. Double Exponential
                amp = y1.max() - y1.min()
                guess_dbl = [0.7 * amp, 0.05, 0.3 * amp, 0.005, y1.min()]
                p_dbl, err_dbl = fit_model(double_exponential, t1, y1, guess_dbl)

                if p_dbl is not None:
                    plt.plot(t1, double_exponential(t1, *p_dbl), color='darkgreen', linestyle='--', linewidth=2,
                             label='Range 1: Dbl Exp Fit')

                    # Calculate t_half for both components
                    t_half_1 = np.log(2) / p_dbl[1]
                    t_half_2 = np.log(2) / p_dbl[3]

                    print(f"  [Dbl Exp Fit]")
                    print(f"     A1    = {p_dbl[0]:.2f}, k1 = {p_dbl[1]:.5f}")
                    print(f"     -> t_1/2 (1) = {t_half_1:.4f} s")
                    print(f"     A2    = {p_dbl[2]:.2f}, k2 = {p_dbl[3]:.5f}")
                    print(f"     -> t_1/2 (2) = {t_half_2:.4f} s")

        # --- RANGE 2: Sigmoidal ---
        if range2[0] is not None and range2[1] is not None:
            mask2 = (fit_source_time >= range2[0]) & (fit_source_time <= range2[1])
            t2 = fit_source_time[mask2]
            y2 = fit_source_int[mask2]

            if len(t2) > 0:
                print(f"\n--- Range 2 ({range2[0]}-{range2[1]}s): Sigmoidal ---")
                guess_sig = estimate_initial_params(t2, y2, 'sigmoid')
                p_sig, err_sig = fit_model(sigmoid_model, t2, y2, guess_sig)

                if p_sig is not None:
                    plt.plot(t2, sigmoid_model(t2, *p_sig), color='orange', linewidth=2, label='Range 2: Sigmoid Fit')
                    print(f"  [Sigmoid Fit]")
                    print(f"     Max Growth = {p_sig[1]:.2f}")
                    print(f"     Rate (k)   = {p_sig[2]:.5f}")
                    print(f"     t_half     = {p_sig[3]:.4f} s")

    # -------------------------------------------------------
    # LOGIC: ORIGINAL SINGLE RANGE MODES
    # -------------------------------------------------------
    elif fit_type in ['exponential', 'sigmoid', 'two_models', 'exp_sigmoid', 'double_exp_sigmoid']:
        if range1[0] is not None and range1[1] is not None:
            mask = (fit_source_time >= range1[0]) & (fit_source_time <= range1[1])
            fit_time = fit_source_time[mask]
            fit_intensity = fit_source_int[mask]
        else:
            fit_time = fit_source_time
            fit_intensity = fit_source_int

        if fit_type == 'exponential':
            guess = estimate_initial_params(fit_time, fit_intensity, 'exponential')
            p, e = fit_model(exponential, fit_time, fit_intensity, guess)
            if p is not None: plt.plot(fit_time, exponential(fit_time, *p), 'g-', label='Exp Fit')

        elif fit_type == 'two_models':
            # This is the old "two_models" logic (fit exp and double exp on ONE range)
            guess_exp = estimate_initial_params(fit_time, fit_intensity, 'exponential')
            p_exp, _ = fit_model(exponential, fit_time, fit_intensity, guess_exp)

            amp = fit_intensity.max() - fit_intensity.min()
            guess_dbl = [0.7 * amp, 0.05, 0.3 * amp, 0.005, fit_intensity.min()]
            p_dbl, _ = fit_model(double_exponential, fit_time, fit_intensity, guess_dbl)

            if p_exp is not None: plt.plot(fit_time, exponential(fit_time, *p_exp), 'g-', label='Exp Fit')
            if p_dbl is not None: plt.plot(fit_time, double_exponential(fit_time, *p_dbl), 'purple', linestyle='--',
                                           label='Dbl Exp Fit')

        elif fit_type == 'sigmoid':
            guess = estimate_initial_params(fit_time, fit_intensity, 'sigmoid')
            p, e = fit_model(sigmoid_model, fit_time, fit_intensity, guess)
            if p is not None: plt.plot(fit_time, sigmoid_model(fit_time, *p), 'orange', label='Sigmoid Fit')

        elif fit_type == 'exp_sigmoid':
            guess = estimate_initial_params(fit_time, fit_intensity, 'exp_sigmoid')
            p, e = fit_model(exp_sigmoid_model, fit_time, fit_intensity, guess)
            if p is not None: plt.plot(fit_time, exp_sigmoid_model(fit_time, *p), 'k-', label='Exp+Sig Fit')

    # Final Plot Settings
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Kinetics Analysis: {fit_type}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()
    print(f"Plot saved as {output_plot}")


# ===========================================================
# 5. Execution
# ===========================================================

if __name__ == "__main__":
    # Update this path to your actual file
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F8E4N/kinetics/fluo/F8_september/F8_12h_6.csv"

    # Check if file exists, else create dummy for demo
    try:
        df = read_data(filename)
    except FileNotFoundError:
        print("File not found, generating dummy data for demonstration...")
        t = np.linspace(0, 4000, 4000)
        # Create data that looks like decay then growth
        y = 5000 * np.exp(-0.01 * t) + 1000 + 3000 / (1 + np.exp(-0.01 * (t - 2500)))
        y += np.random.normal(0, 50, size=len(t))
        df = pd.DataFrame({'Time': t, 'Intensity': y})

    out = filename[:-4] + "_fit.png"

    # -------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------
    plot_data(df,
              smooth_method='savitzky_golay',
              window_size=15,
              polyorder=3,
              output_plot=out,
              dead_time=0,

              # ENABLE THE SPLIT MODE
              fit_type="split_regime",

              # Range 1: Decay (Fits Exp + Double Exp)
              range1=(0, 1950),  # Fits Exp and DblExp here

              # Range 2: Growth (Fits Sigmoid)
              range2=(1900, 3000)  # Fits Sigmoid here
              )

