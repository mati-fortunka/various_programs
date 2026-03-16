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

    # Group files by basename (without _chN if present)
    grouped = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".bka"):
            continue
        base = re.sub(r"_ch\d+\.bka$", "", fname)  # strip _chN if present
        grouped.setdefault(base, []).append(fname)

    for base, files in grouped.items():
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

        if not dfs:
            print(f"⚠️ Skipping {base}, no valid data")
            continue

        # If multiple scans, align wavelengths and average
        if len(dfs) > 1:
            merged = dfs[0]
            for df in dfs[1:]:
                merged = pd.merge(merged, df, on="Wavelength", how="inner", suffixes=("", "_dup"))
            # average across all Ellipticity columns
            ellip_cols = [c for c in merged.columns if "Ellipticity" in c]
            merged["Ellipticity"] = merged[ellip_cols].mean(axis=1)
            df_avg = merged[["Wavelength", "Ellipticity"]]
        else:
            df_avg = dfs[0]

        # --- Parse concentration ---
        conc = None

        # Case 1: sav-golay0.2m.bka → decimal M
        match = re.search(r"golay([\d\.]+)m", base, re.IGNORECASE)
        if match:
            conc = float(match.group(1))

        # Case 2: tm45 → 4.5 M (or tm5 → 5.0 M)
        if conc is None:
            match = re.search(r"tm(\d+)", base, re.IGNORECASE)
            if match:
                raw = match.group(1)
                conc = float(raw) / 10.0 if len(raw) > 1 else float(raw)

        if conc is not None:
            spectra[conc] = df_avg
        else:
            print(f"⚠️ Could not parse concentration from {base}")

    return dict(sorted(spectra.items()))


# -------------------
# ANALYSIS + PLOTS
# -------------------
def analyse_two_sets(unfold_folder, refold_folder,
                     wavelength_unfold, wavelength_refold,
                     smoothing_method="savitzky_golay",
                     window_size=15, spline_smoothing_factor=0.5,
                     poly_order=3, baseline_wavelength=None,
                     output_txt="fit_results.txt"):

    save_folder = os.path.dirname(unfold_folder)
    open(output_txt, "w").close()

    def process_set(folder, wavelength, label):
        spectra = load_bka_files(folder)
        ellipticity_vs_conc = []
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

            # --- Order ascending ---
            if wavelengths[0] > wavelengths[-1]:
                wavelengths = wavelengths[::-1]
                smoothed = smoothed[::-1]

            # --- Baseline correction ---
            baseline_value = 0.0
            if baseline_wavelength is not None:
                baseline_value = np.interp(baseline_wavelength, wavelengths, smoothed)

            target_ellipticity = np.interp(wavelength, wavelengths, smoothed)
            corrected_ellipticity = target_ellipticity - baseline_value

            ellipticity_vs_conc.append((conc, corrected_ellipticity))
            all_spectra.append((wavelengths, smoothed, conc))

        plot_data = pd.DataFrame(ellipticity_vs_conc, columns=["den_concentration", "Ellipticity"])
        plot_data.sort_values(by="den_concentration", inplace=True)

        # --- Fit ---
        x_data = plot_data["den_concentration"].values
        y_data = plot_data["Ellipticity"].values
        initial_guess = [min(y_data), max(y_data), 2, 2]

        try:
            popt, pcov = curve_fit(G, x_data, y_data, p0=initial_guess)
            perr = np.sqrt(np.diag(pcov))
            print(f"\n{label} fitted parameters:")
            print(f"a_n = {popt[0]:.3f} ± {perr[0]:.3f}")
            print(f"a_u = {popt[1]:.3f} ± {perr[1]:.3f}")
            print(f"m   = {popt[2]:.3f} ± {perr[2]:.3f}")
            print(f"d   = {popt[3]:.3f} ± {perr[3]:.3f}")

            # Save to text file
            with open(output_txt, "a") as f:
                f.write(f"\n{label} fitted parameters (wavelength {wavelength} nm):\n")
                f.write(f"a_n = {popt[0]:.6f} ± {perr[0]:.6f}\n")
                f.write(f"a_u = {popt[1]:.6f} ± {perr[1]:.6f}\n")
                f.write(f"m   = {popt[2]:.6f} ± {perr[2]:.6f}\n")
                f.write(f"d   = {popt[3]:.6f} ± {perr[3]:.6f}\n")

        except Exception as e:
            print(f"⚠️ Curve fitting failed for {label}: {e}")
            popt = None

        return plot_data, popt, all_spectra

    # --- 1. Procesowanie danych (teraz zapisujemy też all_spectra) ---
    unfold_data, unfold_fit, unfold_spectra = process_set(unfold_folder, wavelength_unfold, "Unfolding")
    refold_data, refold_fit, refold_spectra = process_set(refold_folder, wavelength_refold, "Refolding")

    # --- 2. Wykres dopasowania (ten co był) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.scatter(unfold_data["den_concentration"], unfold_data["Ellipticity"], label=f"Unfolding", color="blue")
    ax1.scatter(refold_data["den_concentration"], refold_data["Ellipticity"], label=f"Refolding", color="red",
                marker="s")
    if unfold_fit is not None:
        fx = np.linspace(unfold_data["den_concentration"].min(), unfold_data["den_concentration"].max(), 200)
        ax1.plot(fx, G(fx, *unfold_fit), color="blue", linestyle="--")
    if refold_fit is not None:
        fx = np.linspace(refold_data["den_concentration"].min(), refold_data["den_concentration"].max(), 200)
        ax1.plot(fx, G(fx, *refold_fit), color="red", linestyle="--")
    ax1.set_title("Unfolding vs Refolding (Titration Curve)")
    ax1.legend()
    fig1.savefig(f"{save_folder}/fit_comparison.png")

    # --- 3. NOWY WYKRES: WSZYSTKIE WIDMA ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    # Znalezienie max stężenia do normalizacji kolorów
    all_concs = [s[2] for s in unfold_spectra] + [s[2] for s in refold_spectra]
    max_c = max(all_concs) if all_concs else 1.0
    norm = Normalize(vmin=0, vmax=max_c)

    cmap_u = plt.get_cmap("Blues")
    cmap_r = plt.get_cmap("Reds")

    # Rysowanie widm Unfolding
    for w, e, c in unfold_spectra:
        # 0.3 offsetu żeby najniższe stężenie nie było całkiem białe
        color = cmap_u(0.2 + 0.9 * norm(c))
        ax2.plot(w, e, color=color, lw=1, alpha=0.8)

    # Rysowanie widm Refolding
    for w, e, c in refold_spectra:
        color = cmap_r(0.2 + 0.9 * norm(c))
        ax2.plot(w, e, color=color, lw=1, alpha=0.8)

    # Dodanie legendy "ręcznie" dla grup
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    ax2.legend(custom_lines, ['Unfolding (light to dark blue)', 'Refolding (light to dark red)'])

    ax2.set_title("All CD Spectra (Intensity ~ Concentration)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Ellipticity (mdeg)")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{save_folder}/all_spectra_comparison.png")

    plt.show()

    return (unfold_data, unfold_fit), (refold_data, refold_fit)


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    path_unf = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/GuCl/tm_unfold"
    path_ref = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/GuCl/tm_refold"
    analyse_two_sets(unfold_folder = path_unf, refold_folder = path_ref,
                     wavelength_unfold=220, wavelength_refold=220, smoothing_method="savitzky_golay",
                     window_size=15, spline_smoothing_factor=0.5,
                     poly_order=3, baseline_wavelength=None,
                     output_txt="fit_results.txt")