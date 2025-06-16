import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.linalg import svd

# === Settings ===
input_folder = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/unfolding_SAV"  # Change to your actual folder path
output_plot = os.path.join(input_folder, "CD_spectra_comparison.png")
svd_plot = os.path.join(input_folder, "SVD_structure_content_vs_concentration.png")
svd_components_plot = os.path.join(input_folder, "SVD_first_6_components.png")
svd_all_contributions_plot = os.path.join(input_folder, "SVD_component_contributions_vs_concentration.png")
svd_explained_plot = os.path.join(input_folder, "SVD_explained_variance.png")

smoothing_window = 5
smoothing_polyorder = 3
baseline_wavelength = 250.0
reverse_list = [-1.0, 1.0]  # Change to -1.0 to invert component signs
plot_spectra_flag = False     # Set False to skip the raw spectra plot

# === Load .bka Files ===
def load_bka_files(folder):
    spectra = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith(".bka"):
            path = os.path.join(folder, fname)
            with open(path, 'r') as f:
                lines = f.readlines()
            try:
                start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
            except StopIteration:
                print(f"⚠️ No _DATA section in {fname}")
                continue
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
            label = os.path.splitext(fname)[0].split('-')[-1]
            try:
                numeric_label = float(label.rstrip('m').lstrip('golay'))
                spectra[numeric_label] = df
            except ValueError:
                print(f"⚠️ Could not parse concentration from filename {fname}")
    return dict(sorted(spectra.items()))

# === CD Spectra Plot ===
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

# === Custom SVD Function ===
def calc_SVD(concentrations, spectra, reverse_l, q=False):
    all_vals_for_svd = []
    for i in spectra:
        all_vals_for_svd.append([k[1] for k in i])
    M = np.array(all_vals_for_svd).T
    U, s, Vh = svd(M)
    n_sing = 1
    for i in range(len(s)):
        n_sing = i
        if round(s[i] / sum(s) * 100, 2) < 5:
            break
    m_size = U.shape[0]
    sigma = np.zeros((m_size, n_sing))
    for i in range(min(m_size, n_sing)):
        sigma[i, i] = s[i]
    Vh_red = Vh[:n_sing, :]
    Mrev3 = np.dot(U, np.dot(sigma, Vh_red))
    SVD_test = None
    for i in range(n_sing):
        weighted = reverse_l[i] * s[i] * Vh_red[i, :]
        if SVD_test is None:
            SVD_test = list(weighted)
        else:
            SVD_test = [a + b for a, b in zip(SVD_test, weighted)]
    return SVD_test, Vh_red, s, U, M

# === SVD Analysis with Custom Function ===
def perform_svd_analysis(spectra_dict, reverse_l_list=[0]):
    concentrations = list(spectra_dict.keys())
    wavelengths = spectra_dict[concentrations[0]]["Wavelength"].values
    spectra = []
    for conc in concentrations:
        df = spectra_dict[conc]
        y = savgol_filter(df["Ellipticity"].values, smoothing_window, smoothing_polyorder)
        baseline_idx = np.argmin(np.abs(df["Wavelength"].values - baseline_wavelength))
        y -= y[baseline_idx]
        spectra.append(list(zip(df["Wavelength"].values, y)))
    if reverse_l_list == [0]:
        reverse_l = [1.0] * len(spectra)
    else:
        reverse_l = reverse_l_list

    SVD_test, Vh_red, s, U, M = calc_SVD(concentrations, spectra, reverse_l, q=False)

    # === Plot total SVD structure signal
    plt.figure(figsize=(7, 5))
    plt.plot(concentrations, SVD_test, marker='o', label='Structure Contribution (Weighted Sum)')
    plt.xlabel("Denaturant Concentration [M]", fontsize=14)
    plt.ylabel("SVD Component (Weighted)", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(svd_plot, dpi=600)
    plt.show()

    # === Plot individual component contributions
    n_sing = Vh_red.shape[0]
    plt.figure(figsize=(8, 6))
    for i in range(n_sing):
        y_vals = reverse_l[i] * s[i] * Vh_red[i, :]
        plt.plot(concentrations, y_vals, marker='o', label=f'Component {i+1}')
    plt.xlabel("Denaturant Concentration [M]", fontsize=14)
    plt.ylabel("Component Contribution", fontsize=14)
    plt.title("SVD Components vs Concentration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(svd_all_contributions_plot, dpi=600)
    plt.show()

    # === Plot explained variance per component
    variance_explained = (s[:n_sing] ** 2) / np.sum(s ** 2)
    component_indices = np.arange(1, n_sing + 1)

    # === Print explained variance per component
    print("\nExplained variance by component:")
    for idx, var in enumerate(variance_explained * 100, 1):
        print(f"  Component {idx}: {var:.2f}%")

    plt.figure(figsize=(7, 5))
    plt.bar(component_indices, variance_explained * 100, color='skyblue')
    plt.xlabel("Component Index", fontsize=14)
    plt.ylabel("Explained Variance [%]", fontsize=14)
    plt.title("SVD Explained Variance by Component", fontsize=14)
    plt.grid(axis='y')
    plt.xticks(component_indices)
    plt.tight_layout()
    plt.savefig(svd_explained_plot, dpi=600)
    plt.show()

    # === Plot spectral shapes (U)
    fig_rows = (n_sing + 1) // 2
    fig, axs = plt.subplots(fig_rows, 2, figsize=(10, 4 * fig_rows), sharex=True)
    axs = axs.flatten()
    for i in range(n_sing):
        axs[i].plot(wavelengths, U[:, i], label=f"Component {i+1}")
        axs[i].legend()
        axs[i].grid(True)
    fig.suptitle(f"First {n_sing} SVD Components (U)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(svd_components_plot, dpi=600)
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    spectra_dict = load_bka_files(input_folder)
    if spectra_dict:
        if plot_spectra_flag:
            plot_cd_spectra(spectra_dict)
        perform_svd_analysis(spectra_dict, reverse_l_list=reverse_list)
    else:
        print("No valid .bka files found in the folder.")
