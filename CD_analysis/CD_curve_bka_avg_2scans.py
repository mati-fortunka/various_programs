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
RT = 0.592  # kcal/mol (RT at ~25°C)


# -------------------
# 2-state model
# -------------------
def G(x, a_n, a_u, m, d):
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


# -------------------
# LOAD & AVERAGE _ch1/_ch2 FILES
# -------------------
def load_bka_files(folder):
    spectra = {}  # concentration -> DataFrame

    # Group files by base name without _ch1/_ch2
    grouped = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".bka"):
            continue
        base = re.sub(r"_ch[12]\.bka$", "", fname)
        grouped.setdefault(base, []).append(fname)

    for base, files in grouped.items():
        if len(files) < 2:
            print(f"⚠️ Only one channel found for {base}, skipping")
            continue

        dfs = []
        for fname in files:
            path = os.path.join(folder, fname)
            with open(path, "r") as f:
                lines = f.readlines()

            # Find the _DATA section
            try:
                start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
            except StopIteration:
                print(f"⚠️ No _DATA section in {fname}")
                continue

            # Extract numeric data
            data = []
            for line in lines[start_idx:]:
                if line.strip() == "" or not any(c.isdigit() for c in line):
                    break
                try:
                    wavelength, value = map(float, line.strip().split())
                    data.append((wavelength, value))
                except ValueError:
                    continue

            if not data:
                print(f"⚠️ No numeric data in {fname}")
                continue

            df = pd.DataFrame(data, columns=["Wavelength", "Ellipticity"])
            dfs.append(df)

        if len(dfs) < 2:
            print(f"⚠️ Skipping {base}, not enough valid channels")
            continue

        # Align wavelengths (use intersection to be safe)
        df1, df2 = dfs
        merged = pd.merge(df1, df2, on="Wavelength", suffixes=("_ch1", "_ch2"))
        merged["Ellipticity"] = merged[["Ellipticity_ch1", "Ellipticity_ch2"]].mean(axis=1)
        df_avg = merged[["Wavelength", "Ellipticity"]]

        # Parse concentration from base name (e.g. tm45 -> 4.5, tm5 -> 5.0)
        match = re.search(r"tm(\d+)", base, re.IGNORECASE)
        if match:
            raw = match.group(1)
            if len(raw) > 1:
                conc = float(raw) / 10.0
            else:
                conc = float(raw)
            spectra[conc] = df_avg
        else:
            print(f"⚠️ Could not parse concentration from {base}")

    return dict(sorted(spectra.items()))


# -------------------
# ANALYSIS + PLOTS
# -------------------
def analyse_bka_and_plot(folder_path, wavelength,
                         smoothing_method="savitzky_golay",
                         window_size=15, spline_smoothing_factor=0.5,
                         poly_order=3, baseline_wavelength=None):
    spectra = load_bka_files(folder_path)

    ellipticity_vs_concentration = []
    all_spectra = []

    for conc, df in spectra.items():
        wavelengths = df["Wavelength"].to_numpy()
        ellipticity = df["Ellipticity"].to_numpy()

        # --- Smoothing ---
        smoothed = ellipticity.copy()
        if smoothing_method == "moving_average":
            if len(ellipticity) >= window_size:
                smoothed = pd.Series(ellipticity).rolling(window=window_size, center=True).mean().to_numpy()
                smoothed[np.isnan(smoothed)] = ellipticity[np.isnan(smoothed)]
        elif smoothing_method == "spline" and len(wavelengths) >= 4:
            spline = UnivariateSpline(wavelengths, ellipticity, s=spline_smoothing_factor)
            smoothed = spline(wavelengths)
        elif smoothing_method == "savitzky_golay":
            if window_size % 2 == 0:
                raise ValueError("Window size for Savitzky-Golay filter must be odd.")
            if len(ellipticity) >= window_size:
                smoothed = savgol_filter(ellipticity, window_length=window_size, polyorder=poly_order)

        # --- Order wavelengths ascending ---
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            smoothed = smoothed[::-1]

        # --- Baseline correction ---
        baseline_value = 0.0
        if baseline_wavelength is not None:
            if baseline_wavelength < wavelengths.min() or baseline_wavelength > wavelengths.max():
                print(f"⚠️ baseline_wavelength {baseline_wavelength} out of range for {conc} M")
            baseline_value = np.interp(baseline_wavelength, wavelengths, smoothed)

        target_ellipticity = np.interp(wavelength, wavelengths, smoothed)
        corrected_ellipticity = target_ellipticity - baseline_value

        ellipticity_vs_concentration.append((conc, corrected_ellipticity))
        all_spectra.append((wavelengths, smoothed, conc))

    # --- Collect results ---
    plot_data = pd.DataFrame(ellipticity_vs_concentration, columns=['den_concentration', 'Ellipticity'])
    plot_data.sort_values(by='den_concentration', inplace=True)

    x_data = plot_data['den_concentration'].values
    y_data = plot_data['Ellipticity'].values

    # --- Curve fitting ---
    initial_guess = [min(y_data), max(y_data), 2, 2]
    try:
        popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        print("Fitted parameters:")
        print(f"a_n = {popt[0]:.3f} ± {perr[0]:.3f}")
        print(f"a_u = {popt[1]:.3f} ± {perr[1]:.3f}")
        print(f"m   = {popt[2]:.3f} ± {perr[2]:.3f}")
        print(f"d   = {popt[3]:.3f} ± {perr[3]:.3f}")
    except Exception as e:
        print(f"⚠️ Curve fitting failed: {e}")
        popt = None

    # --- Scatter + Fit plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(x_data, y_data, label='Data', color='blue', marker='o')
    if popt is not None:
        fit_x = np.linspace(x_data.min(), x_data.max(), 200)
        ax1.plot(fit_x, G(fit_x, *popt), label='Fit', color='red', linestyle='--')
    ax1.set_title(f'Ellipticity at {wavelength} nm vs Denaturant Concentration')
    ax1.set_xlabel('Denaturant Concentration (M)')
    ax1.set_ylabel('Mean Ellipticity (mdeg)')
    ax1.grid(True)
    ax1.legend()
    fig1.savefig(os.path.join(folder_path, f"CD_{wavelength}nm_{smoothing_method}_fit.png"))
    plt.show()

    # --- Combined spectra heatmap-style plot ---
    if all_spectra:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        concs = [c for _, _, c in all_spectra]
        norm = Normalize(vmin=min(concs), vmax=max(concs))
        cmap = cm.viridis

        for wls, sm, conc in all_spectra:
            ax2.plot(wls, sm, color=cmap(norm(conc)), alpha=0.9)

        smap = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig2.colorbar(smap, ax=ax2)
        cbar.set_label("Denaturant concentration (M)")

        ax2.set_title("Smoothed CD Spectra (Heatbar by Denaturant Concentration)")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Ellipticity (mdeg)")
        ax2.grid(True)
        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_path, "combined_cd_spectra_heatbar.png"))
        plt.show()

    return plot_data, popt


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    path = "/home/matifortunka/Documents/JS/data_Cambridge/fusions/fusion_Tm1570_cent/GuCl/tm_refold"
    analyse_bka_and_plot(path, 225,
                         smoothing_method="savitzky_golay",
                         window_size=25, spline_smoothing_factor=0.5,
                         poly_order=3, baseline_wavelength=250)
