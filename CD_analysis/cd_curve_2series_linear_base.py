import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Constants
RT = 0.592  # kcal/mol


# --- 1. MODEL FUNCTIONS ---

def G_linear(x, a_n, slope_n, a_u, slope_u, m, d):
    """
    Santoro-Bolen 2-state model with linear baselines.
    a_n, slope_n: Intercept/Slope for Native state
    a_u, slope_u: Intercept/Slope for Unfolded state
    m: m-value (cooperativity)
    d: Midpoint (Cm)
    """
    y_native = a_n + slope_n * x
    y_unfolded = a_u + slope_u * x
    # DeltaG = m * (d - x)
    # K = exp(-DeltaG / RT) = exp(m * (x - d) / RT)
    exponent = (m * (x - d)) / RT
    # Cap exponent to avoid overflow
    K = np.exp(np.clip(exponent, -500, 500))
    return (y_native + y_unfolded * K) / (1 + K)


# --- 2. DATA PROCESSING ENGINE ---

def process_series(folder_path, target_wl, conc_file, series_name,
                   baseline_wl=250, hv_limit=900, smooth_win=15):
    """
    Processes an entire folder, corrects baselines, fits data, and returns results.
    """
    # Load concentrations
    conc_df = pd.read_csv(conc_file, sep="\t")
    mapping = conc_df.set_index('Sample_number')['den_concentration']

    raw_results = []
    spectra_list = []

    def find_block(lines, prop):
        for i in range(len(lines) - 1):
            if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith(prop):
                start = i + 2
                end = start
                while end < len(lines) and lines[end].strip() and not lines[end].startswith("Wavelength"):
                    end += 1
                return start, end
        return None, None

    # Loop through files
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".csv"): continue
        fpath = os.path.join(folder_path, fname)

        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()

            # Extract CD and HV
            c_s, c_e = find_block(lines, "CircularDichroism")
            h_s, h_e = find_block(lines, "HV")
            if c_s is None: continue

            cd_data = np.array([list(map(float, l.replace(',', ' ').split())) for l in lines[c_s:c_e]])
            wls, ellip = cd_data[:, 0], cd_data[:, 1]

            # HV Filtering (per point)
            if h_s is not None:
                hv_data = np.array([list(map(float, l.replace(',', ' ').split())) for l in lines[h_s:h_e]])
                # Simple mask (assuming same wavelength scale)
                mask = hv_data[:, 1] <= hv_limit
                wls, ellip = wls[mask], ellip[mask]

            # Smoothing
            if len(ellip) > smooth_win:
                ellip = savgol_filter(ellip, window_length=smooth_win, polyorder=3)

            # Ensure ascending wavelength
            if wls[0] > wls[-1]:
                wls, ellip = wls[::-1], ellip[::-1]

            # BASELINE CORRECTION (Applied to the whole spectrum)
            b_val = np.interp(baseline_wl, wls, ellip)
            corrected_spectrum = ellip - b_val

            # Single point for titration curve
            y_point = np.interp(target_wl, wls, corrected_spectrum)

            match = re.search(r"(\d{5})\.csv$", fname)
            if match:
                s_num = int(match.group(1))
                conc = mapping.get(s_num)
                if conc is not None:
                    raw_results.append([conc, y_point])
                    spectra_list.append((wls, corrected_spectrum, conc))
        except Exception as e:
            print(f"Error in {fname}: {e}")

    # Create Dataframe
    df = pd.DataFrame(raw_results, columns=['x', 'y']).sort_values('x')
    x_data, y_data = df['x'].values, df['y'].values

    # --- INITIAL GUESS CALCULATION ---
    # Intercepts: Start and end points
    a_n_init = y_data[0]
    a_u_init = y_data[-1]
    # Slopes: Use first/last 3 points
    sl_n_init = (y_data[2] - y_data[0]) / (x_data[2] - x_data[0]) if len(x_data) > 3 else 0
    sl_u_init = (y_data[-1] - y_data[-3]) / (x_data[-1] - x_data[-3]) if len(x_data) > 3 else 0
    # Midpoint: where y is halfway between native and unfolded
    d_init = x_data[np.argmin(np.abs(y_data - (a_n_init + a_u_init) / 2))]

    p0 = [a_n_init, sl_n_init, a_u_init, sl_u_init, 2.0, d_init]

    try:
        popt, pcov = curve_fit(G_linear, x_data, y_data, p0=p0)
        perr = np.sqrt(np.diag(pcov))  # Standard deviations (errors)

        # --- IMPROVED PRINTING ---
        print(f"\n{'=' * 40}")
        print(f"  FITTED PARAMETERS: {series_name}")
        print(f"{'=' * 40}")
        print(f"Native Intercept (a_n):  {popt[0]:.4f} ± {perr[0]:.4f}")
        print(f"Native Slope (sl_n):    {popt[1]:.4f} ± {perr[1]:.4f}")
        print(f"Unfolded Intercept (a_u): {popt[2]:.4f} ± {perr[2]:.4f}")
        print(f"Unfolded Slope (sl_u):   {popt[3]:.4f} ± {perr[3]:.4f}")
        print(f"m-value (kcal/mol/M):    {popt[4]:.4f} ± {perr[4]:.4f}")
        print(f"Midpoint (Cm) (M):       {popt[5]:.4f} ± {perr[5]:.4f}")
        print(f"{'=' * 40}\n")

    except Exception as e:
        print(f"Fit failed for {series_name}: {e}")
        popt, perr = None, None
    # Plot Spectra (Heatbar) - This now uses baseline corrected spectra
        # Plot Spectra (Heatbar) - Fixed Colorbar Logic
        if spectra_list:
            fig, ax = plt.subplots(figsize=(8, 5))  # Create explicit fig and ax
            norm = Normalize(vmin=min(df['x']), vmax=max(df['x']))

            for w, s, c in spectra_list:
                ax.plot(w, s, color=cm.viridis(norm(c)), alpha=0.6)

            ax.set_title(f"Baseline Corrected Spectra: {series_name}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Ellipticity (mdeg)")

            # Create the colorbar and explicitly tell it to use 'ax' for space
            mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
            fig.colorbar(mappable, ax=ax, label='Conc (M)')

            plt.show()

    return df, popt, perr


# --- 3. MAIN EXECUTION & COMPARISON ---

if __name__ == "__main__":
    # CONFIGURATION
    base = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/"

    # Path setup
    s1_path = os.path.join(base, "seria2_1")
    s2_path = os.path.join(base, "seria2_2")
    wl = 217

    # Process both
    print("Processing Series 1...")
    d1, p1, e1 = process_series(s1_path, wl, os.path.join(s1_path, "concentrations.txt"), "Series 1")

    print("Processing Series 2...")
    d2, p2, e2 = process_series(s2_path, wl, os.path.join(s2_path, "concentrations.txt"), "Series 2")

    # --- THE COMPARISON PLOT ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Series 1 Plot
    ax.scatter(d1['x'], d1['y'], color='royalblue', label='S1 Data', alpha=0.7)
    if p1 is not None:
        x_fit = np.linspace(d1['x'].min(), d1['x'].max(), 200)
        ax.plot(x_fit, G_linear(x_fit, *p1), color='darkblue', lw=2,
                label=f'S1 Fit (Cm={p1[5]:.2f}±{e1[5]:.2f}M)')

    # Series 2 Plot
    ax.scatter(d2['x'], d2['y'], color='tomato', marker='s', label='S2 Data', alpha=0.7)
    if p2 is not None:
        x_fit = np.linspace(d2['x'].min(), d2['x'].max(), 200)
        ax.plot(x_fit, G_linear(x_fit, *p2), color='darkred', lw=2, linestyle='--',
                label=f'S2 Fit (Cm={p2[5]:.2f}±{e2[5]:.2f}M)')

    ax.set_title(f'Comparison of Stability at {wl} nm', fontsize=14)
    ax.set_xlabel('Denaturant Concentration (M)', fontsize=12)
    ax.set_ylabel('Mean Ellipticity (mdeg)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(base, f"CD_{wl}nm_comparison_lfit.png"))
    plt.show()