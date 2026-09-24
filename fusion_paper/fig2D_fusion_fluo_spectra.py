import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def extract_sample_number(name):
    """Extracts numerical sample ID from file name."""
    matches = re.findall(r'(\d+)', name)
    return int(matches[-1]) if matches else None


def load_concentrations(filepath):
    """Loads concentrations accepting comma/dot decimals and whitespace delimiters."""
    try:
        with open(filepath, 'r') as f:
            text = f.read().replace(',', '.')
        conc_df = pd.read_csv(StringIO(text), sep=r'\s+', engine='python')
        conc_df['Sample_number'] = pd.to_numeric(conc_df.iloc[:, 0], errors='coerce')
        conc_df['den_concentration'] = pd.to_numeric(conc_df.iloc[:, 1], errors='coerce')
        conc_df.dropna(subset=['Sample_number', 'den_concentration'], inplace=True)
        return dict(zip(conc_df['Sample_number'].astype(int), conc_df['den_concentration']))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}


def process_spectra_panel(folder_path, config):
    # 1. Load Concentrations
    conc_path = os.path.join(folder_path, "concentrations.txt")
    conc_map = load_concentrations(conc_path)
    if not conc_map:
        print("No valid concentration mapping loaded. Exiting.")
        return

    # 2. Collect and sort CSV files by sample number
    csv_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.csv'):
            sn = extract_sample_number(fname)
            if sn is not None and sn in conc_map:
                csv_files.append((fname, sn))

    csv_files.sort(key=lambda x: conc_map[x[1]])

    plot_data = []

    # 3. Process Spectra
    for fname, sn in csv_files:
        conc = conc_map[sn]
        fpath = os.path.join(folder_path, fname)

        try:
            # Reads first two columns irrespective of exact header strings
            data = pd.read_csv(fpath, header=1, usecols=[0, 1])
            data.columns = ['Wavelength', 'Intensity']
            data['Wavelength'] = pd.to_numeric(data['Wavelength'], errors='coerce')
            data['Intensity'] = pd.to_numeric(data['Intensity'], errors='coerce')
            data.dropna(inplace=True)

            if data.empty:
                continue

            # Filtering window calculation
            step_size = data['Wavelength'].diff().median()
            window_pts = max(3, int(np.ceil(config['window_nm'] / step_size)))
            if window_pts % 2 == 0:
                window_pts += 1

            # Smoothing
            if config['smoothing'] == "savitzky_golay" and len(data) >= window_pts:
                data['Smoothed'] = savgol_filter(data['Intensity'], window_length=window_pts, polyorder=config['poly'])
            elif config['smoothing'] == "moving_average" and len(data) >= window_pts:
                data['Smoothed'] = data['Intensity'].rolling(window=window_pts, center=True).mean()
            elif config['smoothing'] == "spline" and len(data) >= 4:
                spline = UnivariateSpline(data['Wavelength'], data['Intensity'], s=config['spline_s'])
                data['Smoothed'] = spline(data['Wavelength'])
            else:
                data['Smoothed'] = data['Intensity']

            data.dropna(subset=['Smoothed'], inplace=True)

            # Baseline correction
            if config.get('baseline_wl'):
                base_idx = (data['Wavelength'] - config['baseline_wl']).abs().idxmin()
                baseline_val = data.loc[base_idx, 'Smoothed']
                data['Smoothed'] -= baseline_val

            plot_data.append((conc, data['Wavelength'].values, data['Smoothed'].values))

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if not plot_data:
        print("No spectra processed successfully.")
        return

    # Sort strictly by concentration
    plot_data.sort(key=lambda x: x[0])

    # 4. Rendering Panel D
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=300)

    concentrations = [item[0] for item in plot_data]
    cmap = plt.colormaps[config['colormap']]
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))

    for conc, wls, intensities in plot_data:
        ax.plot(wls, intensities, color=cmap(norm(conc)), linewidth=1.6, alpha=0.92)

    # Typography & Axis Formatting
    ax.set_xlabel('Wavelength (nm)', fontsize=16, labelpad=8, fontweight='medium')
    ax.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=16, labelpad=8, fontweight='medium')

    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2, direction='in')
    ax.tick_params(axis='both', which='minor', length=3.5, width=1.0, direction='in')

    # Despine: clean top and right edge removal
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    plt.margins(x=0.02)

    # Colorbar configuration matching panel styling
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, aspect=22)
    cbar.set_label('[GuCl] (M)', fontsize=16, labelpad=10, fontweight='medium')
    cbar.ax.tick_params(labelsize=14, length=4, width=1.1, direction='out')
    cbar.outline.set_linewidth(1.1)

    plt.tight_layout()

    # Save outputs
    output_dir = os.path.join(folder_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_base = os.path.join(output_dir, config['save_name'])

    plt.savefig(f"{save_base}.png", dpi=300)
    plt.savefig(f"{save_base}.pdf", format="pdf")
    print(f"\nPanel D successfully saved:\n  -> {save_base}.png\n  -> {save_base}.pdf")
    print("-" * 50)
    plt.show()


if __name__ == "__main__":
    folder_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/Fuzja/3/2uM/28.07.26/csv"

    config = {
        'smoothing': "savitzky_golay",
        'window_nm': 15,          # Window size in nanometers
        'poly': 3,
        'spline_s': 0.5,
        'baseline_wl': 400,       # Baseline wavelength correction (nm); None to disable
        'colormap': 'plasma',     # Recommended: 'plasma' or 'viridis' (high-contrast, perceptually uniform)
        'save_name': 'Panel_D_fluorescence_spectra'
    }

    process_spectra_panel(folder_path, config)