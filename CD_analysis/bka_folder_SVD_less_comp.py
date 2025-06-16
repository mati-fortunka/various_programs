import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from sklearn.decomposition import TruncatedSVD

# === Settings ===
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/unfolding_SAV"  # Change to your actual folder path
output_plot = os.path.join(input_folder, "CD_spectra_comparison.png")
svd_plot = os.path.join(input_folder, "SVD_structure_content_vs_concentration.png")
svd_components_plot = os.path.join(input_folder, "SVD_first_6_components.png")
n_svd_components = 3  # Change to 2, 3, 4, 5, etc. depending on what you want to analyze
smoothing_window = 5
smoothing_polyorder = 3
baseline_wavelength = 250.0
normalize_svd = False  # Set to True to enable normalization for SVD only
normalize_structure_fractions = False  # Set to False to disable structure fraction normalization
show_components_4_to_6=False

# === Load .bka Files with Different Denaturant Concentrations ===
def load_bka_files(folder):
    spectra = {}  # concentration_label -> pd.DataFrame

    for fname in os.listdir(folder):
        if fname.lower().endswith(".bka"):
            path = os.path.join(folder, fname)
            with open(path, 'r') as f:
                lines = f.readlines()

            # Find the _DATA section (handle both quoted and unquoted versions)
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

            df = pd.DataFrame(data, columns=["Wavelength", "Ellipticity"])
            label = os.path.splitext(fname)[0].split('-')[-1]  # e.g., "1.5m"
            try:
                numeric_label = float(label.rstrip('m').lstrip('golay'))
                spectra[numeric_label] = df
            except ValueError:
                print(f"⚠️ Could not parse concentration from filename {fname}")

    return dict(sorted(spectra.items()))  # Sort by concentration

# === Plot CD Spectra Comparison ===
def plot_cd_spectra(spectra_dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    concentrations = list(spectra_dict.keys())
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    cmap = cm.viridis

    for conc, df in spectra_dict.items():
        y = savgol_filter(df["Ellipticity"].values, smoothing_window, smoothing_polyorder)
        baseline_idx = np.argmin(np.abs(df["Wavelength"].values - baseline_wavelength))
        y -= y[baseline_idx]
        ax.plot(df["Wavelength"], y, label=f"{conc}m", color=cmap(norm(conc)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Denaturant Concentration [M]")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Ellipticity [mdeg]")
    ax.set_title("CD Spectra for Various Denaturant Concentrations")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.show()

# === Perform SVD and Plot Structure Content ===
def perform_svd_analysis(spectra_dict, show_components_4_to_6=False):
    concentrations = list(spectra_dict.keys())
    wavelengths = spectra_dict[concentrations[0]]['Wavelength'].values
    spectra_matrix = []

    for conc in concentrations:
        df = spectra_dict[conc]
        y = savgol_filter(df["Ellipticity"].values, smoothing_window, smoothing_polyorder)
        baseline_idx = np.argmin(np.abs(df["Wavelength"].values - baseline_wavelength))
        y -= y[baseline_idx]
        if normalize_svd:
            y /= np.linalg.norm(y)
        spectra_matrix.append(y)

    spectra_matrix = np.array(spectra_matrix).T  # shape: (wavelengths, samples)

    # === Truncated SVD ===
    svd_model = TruncatedSVD(n_components=n_svd_components)
    components = svd_model.fit_transform(spectra_matrix)  # shape: (wavelengths, n_components)
    VT = svd_model.components_  # shape: (n_components, samples)
    U = components  # Already truncated
    structure_contributions = VT  # Already n_components x n_samples

    df_struct = pd.DataFrame({"Concentration": concentrations})

    if show_components_4_to_6:
        for i in range(n_svd_components):
            df_struct[f"comp{i+1}"] = structure_contributions[i]
    else:
        fractions = np.maximum(structure_contributions[:3], 0)
        if normalize_structure_fractions:
            fractions = fractions / np.sum(fractions, axis=0, keepdims=True)
        for i, label in enumerate(["coil", "alpha", "beta"][:n_svd_components]):
            df_struct[label] = fractions[i]

    # === Plot Structure vs Concentration ===
    fig, ax = plt.subplots(figsize=(7, 5))
    if show_components_4_to_6:
        for i in range(n_svd_components):
            ax.plot(df_struct["Concentration"], df_struct[f"comp{i+1}"],
                    marker='o', label=f"Component {i+1}")
        ax.set_ylabel("Component Weight", fontsize=14)
    else:
        for label in df_struct.columns[1:]:
            ax.plot(df_struct["Concentration"], df_struct[label],
                    marker='o', label=label.capitalize())
        ax.set_ylabel("Secondary structure content", fontsize=14)

    ax.set_xlabel("Denaturant Concentration [M]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(svd_plot, dpi=600)
    plt.show()

    # === Plot Truncated Components (U) ===
    fig_rows = (n_svd_components + 1) // 2
    fig, axs = plt.subplots(fig_rows, 2, figsize=(10, 4 * fig_rows), sharex=True)
    axs = axs.flatten()
    for i in range(n_svd_components):
        axs[i].plot(wavelengths, U[:, i], label=f"Component {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    fig.suptitle(f"First {n_svd_components} SVD Components")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(svd_components_plot, dpi=600)
    plt.show()


# === Main Execution ===
if __name__ == "__main__":
    spectra_dict = load_bka_files(input_folder)
    if spectra_dict:
        plot_cd_spectra(spectra_dict)
        perform_svd_analysis(spectra_dict, show_components_4_to_6=True)
    else:
        print("No valid .bka files found in the folder.")
