import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

RT = 0.592  # kcal/mol at 298.15 K

# --- PUBLICATION STYLING FOR THE TWO TRANSITIONS ---
# Distinct, colorblind-safe colors tied to the left and right axes
TRANSITION_STYLES = {
    'trans1': {
        'color': '#7570B3',       # Muted Orchid / Purple (Fusion identity)
        'marker': 'o',
        'linestyle': '-',
        'label': r'1st transition I$_{310}$/I$_{332}$',
        'ylabel': r'I$_{310}$ / I$_{332}$'
    },
    'trans2': {
        'color': '#D95F02',       # Warm Vermilion / Rust
        'marker': '^',
        'linestyle': '--',
        'label': r'2nd transition I$_{342}$/I$_{330}$',
        'ylabel': r'I$_{342}$ / I$_{330}$'
    }
}


# --- MODEL DEFINITIONS ---
def G(x, a_n, a_u, m, d):
    """Two-state equilibrium unfolding model."""
    return (a_n + a_u * np.exp((m * (x - d)) / RT)) / (1 + np.exp((m * (x - d)) / RT))


def guess_initial_params_2state(x, y):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    a_n = np.mean(y_sorted[:max(1, len(x) // 3)])
    a_u = np.mean(y_sorted[-max(1, len(x) // 3):])
    dy_dx = np.gradient(y_sorted, x_sorted)
    d = x_sorted[np.argmax(np.abs(dy_dx))]
    m = 5 / max(np.ptp(x), 1e-6)
    return [a_n, a_u, m, d]


# --- UTILITIES ---
def calculate_window_points(wavelength_array, nm_interval):
    step_size = wavelength_array.diff().median()
    if pd.isna(step_size) or step_size == 0:
        return 5
    window_points = int(np.ceil(nm_interval / step_size))
    if window_points % 2 == 0:
        window_points += 1
    return max(3, window_points)


def smooth_data(x, y, method, window_nm, spline_smoothing_factor, poly_order):
    window_size = calculate_window_points(x, window_nm)
    if method == "moving_average":
        return pd.Series(y).rolling(window=window_size, center=True).mean().to_numpy()
    elif method == "spline":
        return UnivariateSpline(x, y, s=spline_smoothing_factor)(x)
    elif method == "savitzky_golay":
        return savgol_filter(y, window_length=window_size, polyorder=poly_order)
    return y


def extract_sample_number(name):
    match = re.findall(r'(\d+)', name)
    return int(match[-1]) if match else None


def load_concentrations(filepath, min_conc=None, max_conc=None):
    """Loads sample concentration mapping with min and max concentration boundaries."""
    try:
        try:
            conc_df = pd.read_csv(filepath, sep=r'\s+')
        except Exception:
            conc_df = pd.read_csv(filepath, sep="\t")

        conc_df["Sample_number"] = pd.to_numeric(conc_df["Sample_number"], errors='coerce')
        conc_df["den_concentration"] = pd.to_numeric(conc_df["den_concentration"], errors='coerce')
        conc_df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
        conc_df["Sample_number"] = conc_df["Sample_number"].astype(int)

        if min_conc is not None:
            conc_df = conc_df[conc_df["den_concentration"] >= min_conc]
        if max_conc is not None:
            conc_df = conc_df[conc_df["den_concentration"] <= max_conc]

        return dict(zip(conc_df["Sample_number"], conc_df["den_concentration"]))
    except Exception as e:
        print(f"Failed to load concentrations from {filepath}: {e}")
        return {}


def load_and_preprocess(fname, folder, config):
    try:
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, header=1, usecols=[0, 1], names=["Wavelength", "Intensity"])
        df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors='coerce')
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return None

        df['Smoothed'] = smooth_data(df['Wavelength'], df['Intensity'],
                                     config['smoothing'], config['window'],
                                     config['spline_s'], config['poly'])
        df.dropna(subset=['Smoothed'], inplace=True)

        if config.get('baseline'):
            base_idx = (df['Wavelength'] - config['baseline']).abs().idxmin()
            baseline_val = df.loc[base_idx, 'Smoothed']
            df['Corrected'] = df['Smoothed'] - baseline_val
        else:
            df['Corrected'] = df['Smoothed']

        return df
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None


def extract_ratio_data(folder, conc_map, wl1, wl2, config, label="Series"):
    """Extracts a specific fluorescence ratio for samples in the provided conc_map."""
    ratio_results = []
    print(f"Extracting ratio {wl1}/{wl2} nm for {label}...")

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        sn = extract_sample_number(fname)
        if sn is None or sn not in conc_map:
            continue

        df = load_and_preprocess(fname, folder, config)
        if df is None:
            continue

        try:
            i1 = df.iloc[(df['Wavelength'] - wl1).abs().argsort()[:1]]['Corrected'].values[0]
            i2 = df.iloc[(df['Wavelength'] - wl2).abs().argsort()[:1]]['Corrected'].values[0]
            ratio_results.append((conc_map[sn], i1 / i2))
        except IndexError:
            pass

    df_ratio = pd.DataFrame(ratio_results, columns=['den_concentration', 'Ratio']).sort_values('den_concentration')
    return df_ratio


# --- DUAL Y-AXIS PUBLICATION PLOT ---
def plot_dual_transition_panel(df1, df2, save_name, base_path, do_fit=True):
    fig, ax1 = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    ax2 = ax1.twinx()

    style1 = TRANSITION_STYLES['trans1']
    style2 = TRANSITION_STYLES['trans2']

    handles = []
    labels = []

    # --- DATASET 1 (LEFT AXIS) ---
    if df1 is not None and not df1.empty:
        x1 = df1['den_concentration'].values
        y1 = df1['Ratio'].values

        sc1 = ax1.scatter(x1, y1, color=style1['color'], marker=style1['marker'],
                          s=55, alpha=0.9, edgecolors='none', linewidths=0, zorder=3)
        handles.append(sc1)
        labels.append(style1['label'])

        if do_fit and len(x1) >= 4:
            try:
                p0_1 = guess_initial_params_2state(x1, y1)
                popt1, pcov1 = curve_fit(G, x1, y1, p0=p0_1, maxfev=10000)
                x_fit1 = np.linspace(np.min(x1), np.max(x1), 400)
                y_fit1 = G(x_fit1, *popt1)
                line1, = ax1.plot(x_fit1, y_fit1, color=style1['color'],
                                  linestyle=style1['linestyle'], linewidth=2.2, zorder=2)

                perr1 = np.sqrt(np.diag(pcov1))
                print("\n--- Fit Parameters for Transition 1 (Left Axis: 310/332 nm) ---")
                p_names = ["a_n", "a_u", "m", "C_mid"]
                for p_name, val, err in zip(p_names, popt1, perr1):
                    print(f"  {p_name}: {val:.4f} ± {err:.4f}")
            except Exception as e:
                print(f"Fitting failed for Transition 1: {e}")

    # --- DATASET 2 (RIGHT AXIS) ---
    if df2 is not None and not df2.empty:
        x2 = df2['den_concentration'].values
        y2 = df2['Ratio'].values

        sc2 = ax2.scatter(x2, y2, color=style2['color'], marker=style2['marker'],
                          s=55, alpha=0.9, edgecolors='none', linewidths=0, zorder=3)
        handles.append(sc2)
        labels.append(style2['label'])

        if do_fit and len(x2) >= 4:
            try:
                p0_2 = guess_initial_params_2state(x2, y2)
                popt2, pcov2 = curve_fit(G, x2, y2, p0=p0_2, maxfev=10000)
                x_fit2 = np.linspace(np.min(x2), np.max(x2), 400)
                y_fit2 = G(x_fit2, *popt2)
                line2, = ax2.plot(x_fit2, y_fit2, color=style2['color'],
                                  linestyle=style2['linestyle'], linewidth=2.2, zorder=2)

                perr2 = np.sqrt(np.diag(pcov2))
                print("\n--- Fit Parameters for Transition 2 (Right Axis: 342/330 nm) ---")
                p_names = ["a_n", "a_u", "m", "C_mid"]
                for p_name, val, err in zip(p_names, popt2, perr2):
                    print(f"  {p_name}: {val:.4f} ± {err:.4f}")
            except Exception as e:
                print(f"Fitting failed for Transition 2: {e}")

    # --- AXES TYPOGRAPHY & COLOR HARMONIZATION ---
    # Common X Axis
    ax1.set_xlabel('[GuCl] (M)', fontsize=16, labelpad=8, fontweight='medium')
    ax1.tick_params(axis='x', which='major', labelsize=14, length=6, width=1.2, direction='in')
    ax1.tick_params(axis='x', which='minor', length=3.5, width=1.0, direction='in')

    # Left Y Axis (Transition 1)
    ax1.set_ylabel(style1['ylabel'], fontsize=16, labelpad=8, fontweight='medium', color=style1['color'])
    ax1.tick_params(axis='y', which='major', labelsize=14, length=6, width=1.2,
                    direction='in', colors=style1['color'])
    ax1.tick_params(axis='y', which='minor', length=3.5, width=1.0,
                    direction='in', colors=style1['color'])

    # Right Y Axis (Transition 2)
    ax2.set_ylabel(style2['ylabel'], fontsize=16, labelpad=8, fontweight='medium', color=style2['color'])
    ax2.tick_params(axis='y', which='major', labelsize=14, length=6, width=1.2,
                    direction='in', colors=style2['color'])
    ax2.tick_params(axis='y', which='minor', length=3.5, width=1.0,
                    direction='in', colors=style2['color'])

    # --- SPINES FORMATTING (Clean despine) ---
    # Hide top spine completely on both axes
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Color & thicken left and right borders to match their data series
    ax1.spines['left'].set_color(style1['color'])
    ax1.spines['left'].set_linewidth(1.3)
    ax1.spines['right'].set_visible(False)

    ax2.spines['right'].set_color(style2['color'])
    ax2.spines['right'].set_linewidth(1.3)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    ax1.spines['bottom'].set_linewidth(1.3)

    # Combined single legend
    ax1.legend(
        handles, labels,
        frameon=False,
        fontsize=12,
        loc='lower right',
        handletextpad=0.2,  # Space between marker/line and text label (default: 0.8)
        handlelength=1.0,  # Width reserved for the marker/handle itself (default: 2.0)
        borderpad=0.2  # Internal whitespace around the legend content (default: 0.4)
    )

    plt.tight_layout()

    # Save both vector PDF and 300 DPI PNG
    save_base = os.path.join(base_path, save_name)
    plt.savefig(f"{save_base}.png", dpi=300)
    plt.savefig(f"{save_base}.pdf", format="pdf")
    print(f"\nSaved dual-axis figure panel:\n  -> {save_base}.png\n  -> {save_base}.pdf")
    print("-" * 50)
    plt.show()


def main():
    config = {
        'smoothing': "savitzky_golay",
        'window': 25,
        'spline_s': 0.5,
        'poly': 3,
        'baseline': 400,
        'do_fit': True
    }

    # Data folder containing both the CSV files and concentrations.txt
    data_dir = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/Fuzja/3/2uM/28.07.26/csv"
    conc_file = os.path.join(data_dir, "concentrations.txt")

    # 1. Dataset 1: Transition 1 (0 to 4 M, Ratio 310/332 nm)
    conc_map_trans1 = load_concentrations(conc_file, min_conc=0.0, max_conc=4.0)
    df_trans1 = extract_ratio_data(data_dir, conc_map_trans1, wl1=310, wl2=332,
                                   config=config, label="Transition 1")

    # 2. Dataset 2: Transition 2 (2.4 to 7.5 M, Ratio 342/330 nm)
    conc_map_trans2 = load_concentrations(conc_file, min_conc=2.4, max_conc=7.5)
    df_trans2 = extract_ratio_data(data_dir, conc_map_trans2, wl1=342, wl2=330,
                                   config=config, label="Transition 2")

    # Generate Dual-Axis Plot Panel
    plot_dual_transition_panel(
        df_trans1,
        df_trans2,
        save_name="Fusion_ratio_transitions",
        base_path=data_dir,
        do_fit=config['do_fit']
    )


if __name__ == "__main__":
    main()