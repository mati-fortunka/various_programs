import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.colors import Normalize
from io import StringIO
from matplotlib.cm import ScalarMappable

# === User Settings ===
input_csv = "/home/matifortunka/Documents/JS/data_Cambridge/js/63/63_september/spectra_kin/6/63_native_old_4h_new00001.csv"
native_spectrum_path = None

path = "/".join(input_csv.split('/')[:-1])
output_plot = f"{path}/6_3_spec_kin_fig."

hv_threshold = 1000
smoothing_window = 15
smoothing_polyorder = 3
transpose_data = True
baseline_correction = False
baseline_wavelength = 250.0
dead_time = 30  # seconds
nm_per_sec = 0.4

# Optional wavelength range cutting ===
cut_xaxis = True          # Set to False to disable
xrange = (214, 250)       # Only used if cut_xaxis=True

# === Helpers ===

def gradient_image(ax, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The Axes to draw on.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular, *cmap*, *extent*, and *transform* may be useful.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, interpolation='bicubic', clim=(0, 1),
                   aspect='auto', **kwargs)
    return im

def extract_section(lines, section_name):
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    else:
        raise ValueError(f"Section {section_name} not found")
    section_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        section_lines.append(line)
    return section_lines

# === Load CSV ===
with open(input_csv, 'r') as f:
    lines = f.readlines()

cd_lines = extract_section(lines, "CircularDichroism")
hv_lines = extract_section(lines, "HV")

cd_df = pd.read_csv(StringIO(''.join(cd_lines)), skipinitialspace=True)
hv_df = pd.read_csv(StringIO(''.join(hv_lines)), skipinitialspace=True)
cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
cd_df.dropna(axis=1, how='all', inplace=True)

cd_col_map = {
    float(col): col
    for col in cd_df.columns
    if col != "Wavelength" and not col.startswith("Unnamed")
}
hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)
hv_df.dropna(axis=1, how='all', inplace=True)
hv_col_map = {
    float(col): col
    for col in hv_df.columns
    if col != "Wavelength" and not col.startswith("Unnamed")
}

if transpose_data:
    print("🔄 Transposing CD and HV dataframes (special reshaping)")

    cd_times = pd.to_numeric(cd_df.iloc[:, 0], errors='coerce')
    cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors='coerce')
    cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times).reset_index()
    cd_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    cd_df = cd_df.dropna(subset=['Wavelength'])
    cd_df = cd_df.sort_values('Wavelength').reset_index(drop=True)
    wavelengths = cd_df['Wavelength'].values

    hv_times = pd.to_numeric(hv_df.iloc[:, 0], errors='coerce')
    hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors='coerce')
    hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

    hv_df = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times).reset_index()
    hv_df.rename(columns={'index': 'Wavelength'}, inplace=True)
    hv_df = hv_df.dropna(subset=['Wavelength'])
    hv_df = hv_df.sort_values('Wavelength').reset_index(drop=True)

    common_wavelengths = np.intersect1d(cd_df['Wavelength'].values, hv_df['Wavelength'].values)

    cd_df = cd_df[cd_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)
    hv_df = hv_df[hv_df['Wavelength'].isin(common_wavelengths)].reset_index(drop=True)
    wavelengths = cd_df['Wavelength'].values

    cd_col_map = {float(col): col for col in cd_df.columns if col != 'Wavelength'}
    hv_col_map = {float(col): col for col in hv_df.columns if col != 'Wavelength'}
else:
    cd_col_map = {float(col): col for col in cd_df.columns[1:]}
    hv_col_map = {float(col): col for col in hv_df.columns[1:]}

cd_times = sorted(cd_col_map.keys())
hv_times = sorted(hv_col_map.keys())
wavelengths = cd_df.iloc[:, 0].values

time_shift = dead_time + (baseline_wavelength - np.min(wavelengths)) / nm_per_sec
shifted_cd_times = [t + time_shift for t in cd_times]
shifted_cd_times_hours = [t / 3600 for t in shifted_cd_times]

cmap_unfolding = ["#ff1f00", "#ff6300", "#ffa700", "#ffec00", "#ceff00", "#8aff00",
                  "#46ff00", "#01ff00", "#00ff43", "#00ff87", "#00ffcb", "#00eeff",
                  "#00aaff", "#0066ff"][::-1]

fig, ax = plt.subplots(figsize=(6, 5))

norm = Normalize(vmin=0, vmax=max(shifted_cd_times_hours))
sm = ScalarMappable(norm=norm, cmap=plt.cm.jet)

for idx, (cd_time, shifted_cd_time_hr) in enumerate(zip(cd_times, shifted_cd_times_hours)):
    colname = cd_col_map.get(cd_time)
    if colname is None:
        continue
    cd = cd_df[colname].values
    hv_time = min(hv_times, key=lambda t: abs(t - cd_time))
    hv_col = hv_col_map.get(hv_time)
    hv = hv_df[hv_col].values
    if len(hv) != len(wavelengths):
        hv = hv[:len(wavelengths)]

    mask = hv <= hv_threshold
    x = wavelengths[mask]
    y = cd[mask]

    # === NEW: Apply wavelength cutting ===
    if cut_xaxis and xrange is not None:
        xmin, xmax = xrange
        cut_mask = (x >= xmin) & (x <= xmax)
        x = x[cut_mask]
        y = y[cut_mask]

    if len(y) >= smoothing_window:
        y = savgol_filter(y, window_length=smoothing_window, polyorder=smoothing_polyorder)

    if baseline_correction and len(x) > 0:
        baseline_idx = np.argmin(np.abs(x - baseline_wavelength))
        baseline_val = y[baseline_idx]
        y = y - baseline_val

    if len(x) > 0:  # Ensure we still have data to plot
        ax.plot(x, y, alpha=0.5, linewidth=2, c=sm.to_rgba(shifted_cd_time_hr))

ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("Wavelength [nm]", fontsize=16)
ax.set_ylabel("Ellipticity [mdeg]", fontsize=16)

if cut_xaxis and xrange is not None:
    ax.set_xlim(xrange)

# === Vertical Gradient Legend (no border, smaller, top-right) ===
grad_ax = fig.add_axes([0.93, 0.15, 0.02, 0.2])
gradient = np.linspace(1, 0, 256).reshape(-1, 1)
grad_ax.imshow(gradient, aspect='auto', cmap=plt.cm.jet)
grad_ax.set_frame_on(False)
ax.tick_params(axis='both', labelsize=15)

min_hr = 0
max_hr = int(round(max(shifted_cd_times_hours)))
grad_ax.set_yticks([0, 255])
grad_ax.set_yticklabels([f"{max_hr} h", f"{min_hr} h"], fontsize=15)
grad_ax.set_xticks([])
grad_ax.tick_params(length=0)

fig.set_constrained_layout(True)
plt.savefig(output_plot+"svg", format='svg', dpi=600, bbox_inches='tight')
plt.savefig(output_plot+"png", format='png', dpi=600, bbox_inches='tight')
plt.show()
