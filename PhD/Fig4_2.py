import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. Konfiguracja czcionki głównej i matematycznej
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"  # zwykła kursywa (do legend: \mathit)
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"  # pogrubienie proste (\mathbf)
mpl.rcParams["mathtext.sf"] = (
    "DejaVu Sans:bold:italic"  # POGRUBIONA KURSYWA (\mathsf)
)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"  # WYCISZA OSTRZEŻENIE O 'CURSIVE'!
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

# ==============================================================================
# 1. Konfiguracja Ścieżek, Parametrów i Palety Barw
# ==============================================================================
FIGURE_ID = "Fig_4.2"
TIME_H = 12
CONC_UM = 2.0
PATH_LENGTH_MM = 1.0
RT = 0.592  # kcal/mol przy 298.15 K

RESIDUES = {
    "TrmD": 262,
    "Tm1570": 224
}

# Ścieżki do danych CD (format ProData)
DIR_CD_TRMD = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/trmd/3/2uM/28_07_26"
DIR_CD_TM1570 = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/tm1570/3/2uM/28_07_26"

# Ścieżki do danych Fluorescencji (format Spektrofluorymetr CSV)
DIR_FLUO_TRMD = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/TrmD/3/2uM/28.07.26/csv"
DIR_FLUO_TM1570 = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/Tm1570/3/2uM/28.07.26/csv"

OUTPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV"
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "Figura_4_2_Izolowane_Domeny.png")
CSV_OUTPUT_PATH = "tables/thermo_fits_summary.csv"

# Master Palette projektu
COLOR_TRMD = "#1F77B4"    # Classic Marine Blue
COLOR_TM1570 = "#FF7F0E"  # Deep Terracotta / Orange

# Obcięta mapa plasma (0.05 - 0.85 eliminuje wyblakły żółty na białym tle)
plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)


# ==============================================================================
# 2. Model Termodynamiczny i Eksport Parametrów do CSV
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Dwustanowy model Santoro-Bolen: N <-> U."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -60, 60))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def export_fit_row(fig_id, construct, probe, time_h, conc_uM, model_name,
                   cm_val, cm_se, m_val, m_se, r2_val):
    dG_val = (m_val * cm_val) if (m_val is not None and cm_val is not None) else None
    row = {
        "Figure": fig_id,
        "Construct": construct,
        "Probe": probe,
        "Condition_Time": f"{time_h}h",
        "Condition_Conc_uM": conc_uM,
        "Model": model_name,
        "Cm_M": round(cm_val, 3) if cm_val is not None else None,
        "Cm_SE": round(cm_se, 3) if cm_se is not None else None,
        "m_kcal_mol_M": round(m_val, 3) if m_val is not None else None,
        "m_SE": round(m_se, 3) if m_se is not None else None,
        "dG_H2O_kcal_mol": round(dG_val, 2) if dG_val is not None else None,
        "R_squared": round(r2_val, 4) if r2_val is not None else None
    }
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(CSV_OUTPUT_PATH):
        df_row.to_csv(CSV_OUTPUT_PATH, index=False)
    else:
        df_row.to_csv(CSV_OUTPUT_PATH, mode="a", header=False, index=False)

    print(f"[{fig_id}] {construct} | {probe}: Cm={cm_val:.2f} ± {cm_se:.2f} M | m={m_val:.2f} ± {m_se:.2f} | R²={r2_val:.4f}")


# ==============================================================================
# 3. Parsowanie Danych CD (ProData) i Konwersja do [theta]_MRW
# ==============================================================================
def mdeg_to_mre(mdeg, residues, conc_uM=CONC_UM, path_mm=PATH_LENGTH_MM):
    """Przelicza mdeg na [theta]_MRW [deg * cm^2 * dmol^-1]."""
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def load_concentrations_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().replace(',', '.')
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine='python')
    df['Sample_number'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    df['den_concentration'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df.dropna(subset=['Sample_number', 'den_concentration'], inplace=True)
    return dict(zip(df['Sample_number'].astype(int), df['den_concentration'].astype(float)))


def load_cd_spectra_prodata(folder_path, residues, hv_cutoff=900.0):
    conc_file = os.path.join(folder_path, "concentrations.txt")
    conc_map = load_concentrations_file(conc_file)
    spectra_by_conc = {}
    pattern = re.compile(r"(\d{5})\.csv$")

    for fname in sorted(os.listdir(folder_path)):
        m = pattern.search(fname)
        if not m:
            continue
        sn = int(m.group(1))
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        cd_idx = next((i for i in range(len(lines) - 1)
                       if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism")), None)
        if cd_idx is None:
            continue

        cd_lines = []
        for ln in lines[cd_idx + 2:]:
            if not ln.strip() or ln.strip().startswith("Wavelength"):
                break
            cd_lines.append(list(map(float, ln.replace(',', ' ').split())))
        cd_arr = np.array(cd_lines)
        wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

        # Filtracja po sygnale fotopowielacza (HV)
        hv_idx = next((i for i in range(len(lines) - 1)
                       if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")), None)
        if hv_idx is not None:
            hv_lines = []
            for ln in lines[hv_idx + 2:]:
                if not ln.strip() or ln.strip().startswith("Wavelength"):
                    break
                hv_lines.append(list(map(float, ln.replace(',', ' ').split())))
            hv_arr = np.array(hv_lines)
            if len(hv_arr) == len(wls):
                mask = hv_arr[:, 1] <= hv_cutoff
                wls, mdeg = wls[mask], mdeg[mask]

        if len(wls) < 15:
            continue

        s_idx = np.argsort(wls)
        wls, mdeg = wls[s_idx], mdeg[s_idx]

        if len(mdeg) >= 11:
            mdeg = savgol_filter(mdeg, 11, 3)

        mre = mdeg_to_mre(mdeg, residues)
        spectra_by_conc[float(conc)] = (wls, mre)

    return spectra_by_conc


# ==============================================================================
# 4. Parsowanie Danych Fluorescencji
# ==============================================================================
def load_fluo_spectra(folder_path):
    conc_file = os.path.join(folder_path, "concentrations.txt")
    conc_map = load_concentrations_file(conc_file)
    spectra_by_conc = {}

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.csv'):
            continue
        matches = re.findall(r'(\d+)', fname)
        if not matches:
            continue
        sn = int(matches[-1])
        conc = conc_map.get(sn, None)
        if conc is None:
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath, header=1, usecols=[0, 1])
            df.columns = ['Wavelength', 'Intensity']
            df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
            df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
            df.dropna(inplace=True)
            if df.empty:
                continue

            step = df['Wavelength'].diff().median()
            w_pts = max(3, int(np.ceil(15.0 / step))) if step > 0 else 11
            w_pts = w_pts + 1 if w_pts % 2 == 0 else w_pts

            if len(df) >= w_pts:
                df['Intensity'] = savgol_filter(df['Intensity'], w_pts, 3)

            # Korekta baseline na 400 nm
            b_idx = (df['Wavelength'] - 400.0).abs().idxmin()
            df['Intensity'] -= df.loc[b_idx, 'Intensity']

            spectra_by_conc[float(conc)] = (df['Wavelength'].values, df['Intensity'].values)
        except Exception:
            continue

    return spectra_by_conc


# ==============================================================================
# 5. Generator Figury 4.2 (Siatka 3 x 2)
# ==============================================================================
def generate_figure_4_2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True, dpi=300)

    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]
    ax_e = axes[2, 0]
    ax_f = axes[2, 1]

    # Wczytanie serii widm
    cd_trmd = load_cd_spectra_prodata(DIR_CD_TRMD, RESIDUES["TrmD"])
    cd_tm1570 = load_cd_spectra_prodata(DIR_CD_TM1570, RESIDUES["Tm1570"])
    fluo_trmd = load_fluo_spectra(DIR_FLUO_TRMD)
    fluo_tm1570 = load_fluo_spectra(DIR_FLUO_TM1570)

    concs_cd_trmd = np.array(sorted(cd_trmd.keys()))
    concs_cd_tm1570 = np.array(sorted(cd_tm1570.keys()))
    concs_fl_trmd = np.array(sorted(fluo_trmd.keys()))
    concs_fl_tm1570 = np.array(sorted(fluo_tm1570.keys()))

    # --------------------------------------------------------------------------
    # PANEL A: Krzywe denaturacji CD przy 217 nm (TrmD vs Tm1570)
    # --------------------------------------------------------------------------
    y_cd_trmd = np.array([np.interp(217.0, cd_trmd[c][0], cd_trmd[c][1]) for c in concs_cd_trmd])
    y_cd_tm1570 = np.array([np.interp(217.0, cd_tm1570[c][0], cd_tm1570[c][1]) for c in concs_cd_tm1570])

    # Fit 2-state TrmD (CD)
    p0_cd_trmd = [y_cd_trmd[0], y_cd_trmd[-1], 2.0, np.median(concs_cd_trmd)]
    popt_cd_trmd, pcov_cd_trmd = curve_fit(two_state_model, concs_cd_trmd, y_cd_trmd, p0=p0_cd_trmd, maxfev=10000)
    perr_cd_trmd = np.sqrt(np.diag(pcov_cd_trmd))
    r2_cd_trmd = calc_r_squared(y_cd_trmd, two_state_model(concs_cd_trmd, *popt_cd_trmd))
    export_fit_row(FIGURE_ID, "CnTrmD", "CD_217nm", TIME_H, CONC_UM, "2-state",
                   popt_cd_trmd[3], perr_cd_trmd[3], popt_cd_trmd[2], perr_cd_trmd[2], r2_cd_trmd)

    # Fit 2-state Tm1570 (CD)
    p0_cd_tm1570 = [y_cd_tm1570[0], y_cd_tm1570[-1], 2.0, np.median(concs_cd_tm1570)]
    popt_cd_tm1570, pcov_cd_tm1570 = curve_fit(two_state_model, concs_cd_tm1570, y_cd_tm1570, p0=p0_cd_tm1570, maxfev=10000)
    perr_cd_tm1570 = np.sqrt(np.diag(pcov_cd_tm1570))
    r2_cd_tm1570 = calc_r_squared(y_cd_tm1570, two_state_model(concs_cd_tm1570, *popt_cd_tm1570))
    export_fit_row(FIGURE_ID, "CnTm1570", "CD_217nm", TIME_H, CONC_UM, "2-state",
                   popt_cd_tm1570[3], perr_cd_tm1570[3], popt_cd_tm1570[2], perr_cd_tm1570[2], r2_cd_tm1570)

    # Punkty i krzywe
    ax_a.scatter(concs_cd_trmd, y_cd_trmd, color=COLOR_TRMD, marker='o', s=55, edgecolors='none', zorder=3,
                 label=r"$Cn$TrmD")
    ax_a.scatter(concs_cd_tm1570, y_cd_tm1570, color=COLOR_TM1570, marker='s', s=50, edgecolors='none', zorder=3,
                 label=r"$Cn$Tm1570")

    x_dense_a_trmd = np.linspace(concs_cd_trmd.min(), concs_cd_trmd.max(), 300)
    x_dense_a_tm = np.linspace(concs_cd_tm1570.min(), concs_cd_tm1570.max(), 300)
    ax_a.plot(x_dense_a_trmd, two_state_model(x_dense_a_trmd, *popt_cd_trmd), color=COLOR_TRMD, lw=2.2, zorder=2)
    ax_a.plot(x_dense_a_tm, two_state_model(x_dense_a_tm, *popt_cd_tm1570), color=COLOR_TM1570, lw=2.2, zorder=2)

    ax_a.set_xlim(-0.2, 7.8)
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=12)
    ax_a.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_a.tick_params(labelsize=10.5)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower right", bbox_to_anchor=(0.96, 0.05), fontsize=10, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_a.text(0.04, 0.95, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_a.set_title("CD Equilibrium Unfolding (217 nm)", fontsize=12.5, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL B: Krzywe denaturacji Fluorescencji przy 335 nm (TrmD vs Tm1570)
    # --------------------------------------------------------------------------
    y_fl_trmd = np.array([np.interp(335.0, fluo_trmd[c][0], fluo_trmd[c][1]) for c in concs_fl_trmd])
    y_fl_tm1570 = np.array([np.interp(335.0, fluo_tm1570[c][0], fluo_tm1570[c][1]) for c in concs_fl_tm1570])

    # Fit 2-state TrmD (Fluo)
    p0_fl_trmd = [y_fl_trmd[0], y_fl_trmd[-1], 2.0, np.median(concs_fl_trmd)]
    popt_fl_trmd, pcov_fl_trmd = curve_fit(two_state_model, concs_fl_trmd, y_fl_trmd, p0=p0_fl_trmd, maxfev=10000)
    perr_fl_trmd = np.sqrt(np.diag(pcov_fl_trmd))
    r2_fl_trmd = calc_r_squared(y_fl_trmd, two_state_model(concs_fl_trmd, *popt_fl_trmd))
    export_fit_row(FIGURE_ID, "CnTrmD", "Fluo_335nm", TIME_H, CONC_UM, "2-state",
                   popt_fl_trmd[3], perr_fl_trmd[3], popt_fl_trmd[2], perr_fl_trmd[2], r2_fl_trmd)

    # Fit 2-state Tm1570 (Fluo)
    p0_fl_tm1570 = [y_fl_tm1570[0], y_fl_tm1570[-1], 2.0, np.median(concs_fl_tm1570)]
    popt_fl_tm1570, pcov_fl_tm1570 = curve_fit(two_state_model, concs_fl_tm1570, y_fl_tm1570, p0=p0_fl_tm1570, maxfev=10000)
    perr_fl_tm1570 = np.sqrt(np.diag(pcov_fl_tm1570))
    r2_fl_tm1570 = calc_r_squared(y_fl_tm1570, two_state_model(concs_fl_tm1570, *popt_fl_tm1570))
    export_fit_row(FIGURE_ID, "CnTm1570", "Fluo_335nm", TIME_H, CONC_UM, "2-state",
                   popt_fl_tm1570[3], perr_fl_tm1570[3], popt_fl_tm1570[2], perr_fl_tm1570[2], r2_fl_tm1570)

    # Punkty i krzywe
    ax_b.scatter(concs_fl_trmd, y_fl_trmd, color=COLOR_TRMD, marker='o', s=55, edgecolors='none', zorder=3,
                 label=r"$Cn$TrmD")
    ax_b.scatter(concs_fl_tm1570, y_fl_tm1570, color=COLOR_TM1570, marker='s', s=50, edgecolors='none', zorder=3,
                 label=r"$Cn$Tm1570")

    x_dense_b_trmd = np.linspace(concs_fl_trmd.min(), concs_fl_trmd.max(), 300)
    x_dense_b_tm = np.linspace(concs_fl_tm1570.min(), concs_fl_tm1570.max(), 300)
    ax_b.plot(x_dense_b_trmd, two_state_model(x_dense_b_trmd, *popt_fl_trmd), color=COLOR_TRMD, lw=2.2, zorder=2)
    ax_b.plot(x_dense_b_tm, two_state_model(x_dense_b_tm, *popt_fl_tm1570), color=COLOR_TM1570, lw=2.2, zorder=2)

    ax_b.set_xlim(-0.2, 7.8)
    ax_b.set_xlabel("[GdmCl] (M)", fontsize=12)
    ax_b.set_ylabel(r"Fluorescence Intensity $I_{335}$ (a.u.)", fontsize=12)
    ax_b.tick_params(labelsize=10.5)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.96, 0.95), fontsize=10, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_b.text(0.04, 0.95, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_b.set_title("Fluorescence Unfolding (335 nm)", fontsize=12.5, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL C: Widma CD dla CnTrmD (200 - 250 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_cd_trmd = Normalize(vmin=concs_cd_trmd.min(), vmax=concs_cd_trmd.max())
    for c in concs_cd_trmd:
        w, mre = cd_trmd[c]
        mask = (w >= 210.0) & (w <= 250.0)
        ax_c.plot(w[mask], mre[mask], color=plasma_trimmed(norm_cd_trmd(c)), lw=1.35)

    ax_c.set_xlim(210.0, 250.0)
    ax_c.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_c.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_c.tick_params(labelsize=10.5)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.text(0.04, 0.95, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_c.set_title(r"$\mathsf{Cn}$TrmD – Far-UV CD Spectra", fontweight="bold", fontsize=12.5, pad=8)

    # Inset colorbar w prawym dolnym rogu
    cax_c = ax_c.inset_axes([0.82, 0.08, 0.035, 0.38])
    sm_c = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd_trmd)
    sm_c.set_array([])
    cbar_c = fig.colorbar(sm_c, cax=cax_c, orientation="vertical")
    cbar_c.ax.tick_params(labelsize=8.5)
    cbar_c.set_label("[GdmCl] (M)", fontsize=8.5, labelpad=5)

    # --------------------------------------------------------------------------
    # PANEL D: Widma Fluo dla CnTrmD (300 - 400 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_fl_trmd = Normalize(vmin=concs_fl_trmd.min(), vmax=concs_fl_trmd.max())
    for c in concs_fl_trmd:
        w, inten = fluo_trmd[c]
        mask = (w >= 300.0) & (w <= 400.0)
        ax_d.plot(w[mask], inten[mask], color=plasma_trimmed(norm_fl_trmd(c)), lw=1.35)

    ax_d.set_xlim(300.0, 400.0)
    ax_d.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_d.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=12)
    ax_d.tick_params(labelsize=10.5)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.text(0.04, 0.95, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_d.set_title(r"$\mathsf{Cn}$TrmD – Fluorescence Emission Spectra", fontweight="bold", fontsize=12.5, pad=8)

     # Inset colorbar w prawym górnym rogu
    cax_d = ax_d.inset_axes([0.82, 0.54, 0.035, 0.38])
    sm_d = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fl_trmd)
    sm_d.set_array([])
    cbar_d = fig.colorbar(sm_d, cax=cax_d, orientation="vertical")
    cbar_d.ax.tick_params(labelsize=8.5)
    cbar_d.set_label("[GdmCl] (M)", fontsize=8.5, labelpad=5)

    # --------------------------------------------------------------------------
    # PANEL E: Widma CD dla CnTm1570 (200 - 250 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_cd_tm = Normalize(vmin=concs_cd_tm1570.min(), vmax=concs_cd_tm1570.max())
    for c in concs_cd_tm1570:
        w, mre = cd_tm1570[c]
        mask = (w >= 210.0) & (w <= 250.0)
        ax_e.plot(w[mask], mre[mask], color=plasma_trimmed(norm_cd_tm(c)), lw=1.35)

    ax_e.set_xlim(210.0, 250.0)
    ax_e.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_e.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_e.tick_params(labelsize=10.5)
    ax_e.grid(True, linestyle=":", alpha=0.5)
    ax_e.text(0.04, 0.95, "E", transform=ax_e.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_e.set_title(r"$\mathsf{Cn}$Tm1570 – Far-UV CD Spectra", fontweight="bold", fontsize=12.5, pad=8)

    # Inset colorbar w prawym dolnym rogu
    cax_e = ax_e.inset_axes([0.82, 0.08, 0.035, 0.38])
    sm_e = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd_tm)
    sm_e.set_array([])
    cbar_e = fig.colorbar(sm_e, cax=cax_e, orientation="vertical")
    cbar_e.ax.tick_params(labelsize=8.5)
    cbar_e.set_label("[GdmCl] (M)", fontsize=8.5, labelpad=5)

    # --------------------------------------------------------------------------
    # PANEL F: Widma Fluo dla CnTm1570 (300 - 400 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_fl_tm = Normalize(vmin=concs_fl_tm1570.min(), vmax=concs_fl_tm1570.max())
    for c in concs_fl_tm1570:
        w, inten = fluo_tm1570[c]
        mask = (w >= 300.0) & (w <= 400.0)
        ax_f.plot(w[mask], inten[mask], color=plasma_trimmed(norm_fl_tm(c)), lw=1.35)

    ax_f.set_xlim(300.0, 400.0)
    ax_f.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_f.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=12)
    ax_f.tick_params(labelsize=10.5)
    ax_f.grid(True, linestyle=":", alpha=0.5)
    ax_f.text(0.04, 0.95, "F", transform=ax_f.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_f.set_title(r"$\mathsf{Cn}$Tm1570 – Fluorescence Emission Spectra",fontweight="bold", fontsize=12.5, pad=8)

    # Inset colorbar w prawym górnym rogu
    cax_f = ax_f.inset_axes([0.82, 0.54, 0.035, 0.38])
    sm_f = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fl_tm)
    sm_f.set_array([])
    cbar_f = fig.colorbar(sm_f, cax=cax_f, orientation="vertical")
    cbar_f.ax.tick_params(labelsize=8.5)
    cbar_f.set_label("[GdmCl] (M)", fontsize=8.5, labelpad=5)

    # Zapis figury w formatach PNG (300 DPI) i wektorowym PDF
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=300)
    pdf_path = os.path.splitext(OUTPUT_FIGURE_PATH)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf")

    print(f"\n[✓] Gotowa Figura 4.2 została zapisana do:\n  -> {OUTPUT_FIGURE_PATH}\n  -> {pdf_path}")
    print(f"[✓] Zbiorcze parametry dopasowań zaktualizowano w: {CSV_OUTPUT_PATH}")
    # plt.show()


if __name__ == "__main__":
    generate_figure_4_2()