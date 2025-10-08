import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import re


# Define the models for curve fitting
# -----------------------------------------------------------

def exponential(t, A, k, c):
    """
    Single exponential decay model.
    """
    return A * np.exp(-k * t) + c


def single_exponential_with_drift(t, A, k, c, m):
    """
    Single exponential decay with a linear drift.
    """
    return A * np.exp(-k * t) + c + m * t


def double_exponential(t, A1, k1, A2, k2, c):
    """
    Double exponential decay model.
    """
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def sigmoid_model(t, y0, a, k, t_half):
    """
    Sigmoid growth model (a variant of the logistic function).
    y0: initial value
    a: maximum growth
    k: growth rate
    t_half: time of half-maximum growth
    """
    return y0 + a / (1 + np.exp(-k * (t - t_half)))


def exp_sigmoid_model(t, A_exp, k_exp, c_exp, y0_sig, a_sig, k_sig, t_half_sig):
    """
    Combines an exponential decay and a sigmoid growth model.
    """
    return exponential(t, A_exp, k_exp, c_exp) + sigmoid_model(t, y0_sig, a_sig, k_sig, t_half_sig)


def double_exp_sigmoid_model(t, A1_exp, k1_exp, A2_exp, k2_exp, c_exp, y0_sig, a_sig, k_sig, t_half_sig):
    """
    Combines a double exponential decay and a sigmoid growth model.
    """
    return double_exponential(t, A1_exp, k1_exp, A2_exp, k2_exp, c_exp) + sigmoid_model(t, y0_sig, a_sig, k_sig,
                                                                                        t_half_sig)


# Data processing functions
# -----------------------------------------------------------

def read_data(filename):
    """
    Reads data from a CSV file, handling headers and potential data wraps.
    """
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        if not re.match(r'^[\d\.\-]', first_line.split(',')[0]):
            print(f"Header detected: '{first_line}'")
            if "Time (s)" in first_line:
                print("\u2705 Time units confirmed as seconds (s).")
            else:
                print("\u26a0\ufe0f Warning: Time units not clearly specified as seconds (s).")
            skiprows = 1
        else:
            print("\u26a0\ufe0f No header detected or malformed header. Assuming data starts immediately.")
            skiprows = 0

    skiprows = 2

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",")
    df_full = df_full.dropna(how='all', axis=1)

    time_col = df_full.iloc[:, 0]
    initial_time = time_col.iloc[0]

    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()
    if wrap_index > 1:
        print(f"Detected secondary block starting at row {wrap_index}. Truncating data.")
        df = df_full.iloc[:wrap_index]
    else:
        df = df_full

    return df


def moving_average(data, window_size):
    """
    Applies a moving average filter to the data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# Helper function for initial parameter estimation
# -----------------------------------------------------------

def estimate_initial_params(time, intensity, model_name):
    """
    Provides initial parameter guesses for different models.
    """
    if model_name == 'exponential':
        A0 = intensity.iloc[0] - intensity.iloc[-1]
        C = intensity.iloc[-1]
        t_half_idx = np.abs(intensity - (C + A0 / 2)).argmin()
        k0 = np.log(2) / time.iloc[t_half_idx] if t_half_idx > 0 else 0.01
        return [A0, k0, C]

    elif model_name == 'sigmoid':
        y0_est = intensity.min()
        a_est = intensity.max() - y0_est
        t_half_idx = np.abs(intensity - (y0_est + a_est / 2)).argmin()
        t_half_est = time.iloc[t_half_idx]
        k_est = 1.0  # A reasonable starting point for k
        return [y0_est, a_est, k_est, t_half_est]

    elif model_name == 'exp_sigmoid':
        exp_params = estimate_initial_params(time, intensity, 'exponential')
        sig_params = estimate_initial_params(time, intensity, 'sigmoid')
        return exp_params + sig_params

    elif model_name == 'double_exp_sigmoid':
        # Assuming double exponential and sigmoid
        A0 = intensity.iloc[0] - intensity.iloc[-1]
        C = intensity.iloc[-1]
        double_exp_params = [0.7 * A0, 0.01, 0.3 * A0, 0.001, C]
        sig_params = estimate_initial_params(time, intensity, 'sigmoid')
        return double_exp_params + sig_params

    return None


# Fitting functions
# -----------------------------------------------------------

def fit_model(model_func, time, intensity, initial_guess):
    """
    Generic function to fit a model and handle errors.
    """
    try:
        popt, pcov = curve_fit(model_func, time, intensity, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print(f"Fit failed for model: {model_func.__name__}")
        return None, None


# Main plotting and fitting function
# -----------------------------------------------------------

def plot_data(df, smooth_method=None, window_size=5, polyorder=2,
              output_plot="out.png", dead_time=20,
              fit_type='exponential', fit_start=None, fit_end=None):
    time = df.iloc[:, 0] + dead_time
    intensity = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(time, intensity, label='Raw Data', color='gray', alpha=0.6)

    if smooth_method == 'moving_average':
        smoothed = moving_average(intensity, window_size)
        time_adjusted = time[:len(smoothed)]
        plt.plot(time_adjusted, smoothed, label=f'Moving Avg (window={window_size})', color='blue')
    elif smooth_method == 'savitzky_golay':
        smoothed = savgol_filter(intensity, window_size, polyorder)
        plt.plot(time, smoothed, label=f'Savitzky-Golay (window={window_size}, poly={polyorder})', color='red')

    if fit_start is not None and fit_end is not None:
        mask = (time >= fit_start) & (time <= fit_end)
        fit_time = time[mask]
        fit_intensity = intensity[mask]
    else:
        fit_time = time
        fit_intensity = intensity

    # Handle single model fitting
    if fit_type == 'exponential':
        initial_guess = estimate_initial_params(fit_time, fit_intensity, 'exponential')
        params, errors = fit_model(exponential, fit_time, fit_intensity, initial_guess)
        if params is not None:
            plt.plot(fit_time, exponential(fit_time, *params), label='Exponential Fit', color='green')
            print("Exponential fit parameters:")
            print(f"  A = {params[0]:.5f} \u00B1 {errors[0]:.5f}")
            print(f"  k = {params[1]:.5f} \u00B1 {errors[1]:.5f}")
            print(f"  c = {params[2]:.5f} \u00B1 {errors[2]:.5f}")
            print(f"  t_1/2 = {np.log(2) / params[1]:.5f}")

    elif fit_type == 'sigmoid':
        initial_guess = estimate_initial_params(fit_time, fit_intensity, 'sigmoid')
        params, errors = fit_model(sigmoid_model, fit_time, fit_intensity, initial_guess)
        if params is not None:
            plt.plot(fit_time, sigmoid_model(fit_time, *params), label='Sigmoid Fit', color='orange')
            print("Sigmoid fit parameters:")
            print(f"  y0 = {params[0]:.5f} \u00B1 {errors[0]:.5f}")
            print(f"  a = {params[1]:.5f} \u00B1 {errors[1]:.5f}")
            print(f"  k = {params[2]:.5f} \u00B1 {errors[2]:.5f}")
            print(f"  t_half = {params[3]:.5f} \u00B1 {errors[3]:.5f}")

    # Handle dual model fitting
    elif fit_type == 'two_models':
        # Fit model 1 (e.g., exponential)
        initial_guess_exp = estimate_initial_params(fit_time, fit_intensity, 'exponential')
        params_exp, errors_exp = fit_model(exponential, fit_time, fit_intensity, initial_guess_exp)

        # Fit model 2 (e.g., double exponential)
        initial_guess_double_exp = [0.7 * fit_intensity.max(), 0.01, 0.3 * fit_intensity.max(), 0.001,
                                    fit_intensity.min()]
        params_double_exp, errors_double_exp = fit_model(double_exponential, fit_time, fit_intensity,
                                                         initial_guess_double_exp)

        if params_exp is not None:
            plt.plot(fit_time, exponential(fit_time, *params_exp), label='Exponential Fit', color='green',
                     linestyle='-')
            print("--- Exponential Fit Parameters ---")
            print(f"  A = {params_exp[0]:.5f} \u00B1 {errors_exp[0]:.5f}")
            print(f"  k = {params_exp[1]:.5f} \u00B1 {errors_exp[1]:.5f}")
            print(f"  c = {params_exp[2]:.5f} \u00B1 {errors_exp[2]:.5f}")
            print(f"  t_1/2 = {np.log(2) / params_exp[1]:.5f}")

        if params_double_exp is not None:
            plt.plot(fit_time, double_exponential(fit_time, *params_double_exp), label='Double Exponential Fit',
                     color='purple', linestyle='--')
            print("\n--- Double Exponential Fit Parameters ---")
            print(f"  A1 = {params_double_exp[0]:.5f} \u00B1 {errors_double_exp[0]:.5f}")
            print(f"  k1 = {params_double_exp[1]:.5f} \u00B1 {errors_double_exp[1]:.5f}")
            print(f"  A2 = {params_double_exp[2]:.5f} \u00B1 {errors_double_exp[2]:.5f}")
            print(f"  k2 = {params_double_exp[3]:.5f} \u00B1 {errors_double_exp[3]:.5f}")
            print(f"  c = {params_double_exp[4]:.5f} \u00B1 {errors_double_exp[4]:.5f}")
            print(f"  t_1 = {np.log(2) / params_double_exp[1]:.5f}")
            print(f"  t_2 = {np.log(2) / params_double_exp[3]:.5f}")

    # Handle combined models
    elif fit_type == 'exp_sigmoid':
        initial_guess = estimate_initial_params(fit_time, fit_intensity, 'exp_sigmoid')
        params, errors = fit_model(exp_sigmoid_model, fit_time, fit_intensity, initial_guess)
        if params is not None:
            plt.plot(fit_time, exp_sigmoid_model(fit_time, *params), label='Exponential + Sigmoid Fit', color='black')
            print("Exponential + Sigmoid fit parameters:")
            print(f"  A_exp = {params[0]:.5f} \u00B1 {errors[0]:.5f}")
            print(f"  k_exp = {params[1]:.5f} \u00B1 {errors[1]:.5f}")
            print(f"  c_exp = {params[2]:.5f} \u00B1 {errors[2]:.5f}")
            print(f"  y0_sig = {params[3]:.5f} \u00B1 {errors[3]:.5f}")
            print(f"  a_sig = {params[4]:.5f} \u00B1 {errors[4]:.5f}")
            print(f"  k_sig = {params[5]:.5f} \u00B1 {errors[5]:.5f}")
            print(f"  t_half_sig = {params[6]:.5f} \u00B1 {errors[6]:.5f}")

    elif fit_type == 'double_exp_sigmoid':
        initial_guess = estimate_initial_params(fit_time, fit_intensity, 'double_exp_sigmoid')
        params, errors = fit_model(double_exp_sigmoid_model, fit_time, fit_intensity, initial_guess)
        if params is not None:
            plt.plot(fit_time, double_exp_sigmoid_model(fit_time, *params), label='Double Exp + Sigmoid Fit',
                     color='darkcyan')
            print("Double Exponential + Sigmoid fit parameters:")
            print(f"  A1_exp = {params[0]:.5f} \u00B1 {errors[0]:.5f}")
            print(f"  k1_exp = {params[1]:.5f} \u00B1 {errors[1]:.5f}")
            print(f"  A2_exp = {params[2]:.5f} \u00B1 {errors[2]:.5f}")
            print(f"  k2_exp = {params[3]:.5f} \u00B1 {errors[3]:.5f}")
            print(f"  c_exp = {params[4]:.5f} \u00B1 {errors[4]:.5f}")
            print(f"  y0_sig = {params[5]:.5f} \u00B1 {errors[5]:.5f}")
            print(f"  a_sig = {params[6]:.5f} \u00B1 {errors[6]:.5f}")
            print(f"  k_sig = {params[7]:.5f} \u00B1 {errors[7]:.5f}")
            print(f"  t_half_sig = {params[8]:.5f} \u00B1 {errors[8]:.5f}")

    # Final plot settings
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence Intensity (a.u.)')
    plt.title('Fluorescence Kinetics Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}")


# --- Main execution part ---

if __name__ == "__main__":
    filename = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/F632/kinetics/fluo/F6_2h_4.csv"
    df = read_data(filename)

    smooth_method = 'savitzky_golay'
    window_size = 15
    polyorder = 3
    dead_time = 0
    out = filename[:-4] + "_fit.png"

    # Example 1: Fitting a single sigmoid model
    # print("--- Fitting Sigmoid Model ---")
    plot_data(df,
              smooth_method=smooth_method,
              window_size=window_size,
              polyorder=polyorder,
              output_plot=out,
              dead_time=dead_time,
              fit_type="sigmoid",
              fit_start=2000,
              fit_end=3000)

    # Example 2: Fitting two models (exponential and double exponential) at once
    # Uncomment the following block to run this example
    # print("\n--- Fitting Two Models ---")
    # plot_data(df,
    #           smooth_method=smooth_method,
    #           window_size=window_size,
    #           polyorder=polyorder,
    #           output_plot=out,
    #           dead_time=dead_time,
    #           fit_type='two_models',
    #           fit_start=None,
    #           fit_end=None)

    # Example 3: Fitting a combined exponential + sigmoidal model
    # Uncomment the following block to run this example
    # print("\n--- Fitting Exponential + Sigmoid Model ---")
    # plot_data(df,
    #           smooth_method=smooth_method,
    #           window_size=window_size,
    #           polyorder=polyorder,
    #           output_plot=out,
    #           dead_time=dead_time,
    #           fit_type='exp_sigmoid',
    #           fit_start=None,
    #           fit_end=None)

    # Example 4: Fitting a combined double exponential + sigmoidal model
    # Uncomment the following block to run this example
    # print("\n--- Fitting Double Exponential + Sigmoid Model ---")
    # plot_data(df,
    #           smooth_method=smooth_method,
    #           window_size=window_size,
    #           polyorder=polyorder,
    #           output_plot=out,
    #           dead_time=dead_time,
    #           fit_type='double_exp_sigmoid',
    #           fit_start=None,
    #           fit_end=None)
