import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Ścieżek, Parametrów i Palety Barw
# ==============================================================================
FIGURE_ID = "Fig_4.3"
CONSTRUCT = "CnTrmD-Tm1570"
TIME_H = 12
CONC_UM = 2.0
RESIDUE_COUNT = 455
PATH_LENGTH_MM = 1.0
RT = 0.592  # kcal/mol przy 298.15 K

# Ścieżki do danych pomiarowych
BASE_DIR_CD = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/3/2uM/28_07_26"
BASE_DIR_FLUO = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry/Fuzja/3/2uM/28.07.26/csv"

CONC_FILE_CD = os.path.join(BASE_DIR_CD, "concentrations.txt")
CONC_FILE_FLUO = os.path.join(BASE_DIR_FLUO, "concentrations.txt")

OUTPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV"
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "Figura_4_3_Fuzja_2uM_12h.png")
CSV_OUTPUT_PATH = "tables/thermo_fits_summary.csv"

# Paleta projektowa
COLOR_FUSION = "#7570B3"  # Fuzja CnTrmD-Tm1570
COLOR_TM1570 = "#D95F02"  # Domena Tm1570 (Ciepły cynober / pomarańcz)
COLOR_TRMD = "#1F77B4"  # Domena TrmD (Marine Blue)

# Utworzenie obciętej palety plasma (zakres 0.05 - 0.85 eliminuje wyblakłą żółć)
plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)


# ==============================================================================
# 2. Modele Termodynamiczne i Moduł Eksportu
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Model dwustanowy Santoro-Bolen: N <-> U."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -60, 60))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def two_state_linear_baseline(x, a_n, b_n, a_u, b_u, d, m):
    """Model dwustanowy z liniowymi liniami bazowymi."""
    dG = m * (d - x)
    dG_clamped = np.clip(dG / RT, -50, 50)
    K = np.exp(-dG_clamped)
    f_u = K / (1.0 + K)
    base_n = a_n + b_n * x
    base_u = a_u + b_u * x
    return base_n + (base_u - base_n) * f_u


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """Model trójstanowy: N <-> I <-> U."""
    sig1 = 1.0 / (1.0 + np.exp(np.clip(-(m1 * (x - d1)) / RT, -60, 60)))
    sig2 = 1.0 / (1.0 + np.exp(np.clip(-(m2 * (x - d2)) / RT, -60, 60)))
    return a_n * (1.0 - sig1) + a_i * (sig1 - sig2) + a_u * sig2


def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def export_fit_row(fig_id, construct, probe, time_h, conc_uM, model_name,
                   cm_val, cm_se, m_val, m_se, r2_val):
    """Zapisuje znormalizowany wiersz parametrów dopasowania do CSV."""
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

    print(f"[{fig_id}] {construct} | {probe} ({model_name}): Cm={cm_val} M, m={m_val}, R²={r2_val:.4f}")


# ==============================================================================
# 3. Wczytywanie Danych CD i Przeliczanie Jednostek
# ==============================================================================
def mdeg_to_mre(mdeg, conc_uM=CONC_UM, path_mm=PATH_LENGTH_MM, residues=RESIDUE_COUNT):
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


def load_prodata_cd(folder_path, conc_file, hv_cutoff=900.0):
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
                       if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith(
            "CircularDichroism")), None)
        if cd_idx is None:
            continue

        cd_lines = []
        for ln in lines[cd_idx + 2:]:
            if not ln.strip() or ln.strip().startswith("Wavelength"):
                break
            cd_lines.append(list(map(float, ln.replace(',', ' ').split())))
        cd_arr = np.array(cd_lines)
        wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

        # Filtracja HV
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

        # Wygładzenie i przeliczenie jednostek
        if len(mdeg) >= 11:
            mdeg = savgol_filter(mdeg, 11, 3)
        mre = mdeg_to_mre(mdeg)
        spectra_by_conc[float(conc)] = (wls, mre)

    return spectra_by_conc


# ==============================================================================
# 4. Wczytywanie Danych Fluorescencji
# ==============================================================================
def load_fluo_spectra(folder_path, conc_file):
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
# 5. Główny Generator Figury 4.3 (2x2)
# ==============================================================================
def generate_figure_4_3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, dpi=300)

    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # --------------------------------------------------------------------------
    # PANEL A: Krzywa rozwijania CD fuzji (217 nm)
    # --------------------------------------------------------------------------
    cd_data = load_prodata_cd(BASE_DIR_CD, CONC_FILE_CD)
    concs_cd = np.array(sorted(cd_data.keys()))

    y_217 = np.array([np.interp(217.0, cd_data[c][0], cd_data[c][1]) for c in concs_cd])

    # Rysowanie punktów doświadczalnych
    ax_a.scatter(
        concs_cd, y_217, color=COLOR_FUSION, marker='^', s=60,
        edgecolors='none', zorder=4, label=f"CnTrmD–Tm1570 (217 nm)"
    )

    x_dense = np.linspace(concs_cd.min(), concs_cd.max(), 300)

    # Dopasowanie 1: Model 2-stanowy (nieudany)
    p0_2s = [y_217[0], y_217[-1], 2.0, np.median(concs_cd)]
    popt_2s, pcov_2s = curve_fit(two_state_model, concs_cd, y_217, p0=p0_2s, maxfev=10000)
    perr_2s = np.sqrt(np.diag(pcov_2s))
    y_fit_2s = two_state_model(x_dense, *popt_2s)
    r2_2s = calc_r_squared(y_217, two_state_model(concs_cd, *popt_2s))

    ax_a.plot(
        x_dense, y_fit_2s, color=COLOR_FUSION, ls="--", lw=2.0,
        label=f"2-state fit ($R^2 = {r2_2s:.3f}$)", zorder=2
    )
    export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm", TIME_H, CONC_UM, "2-state",
                   popt_2s[3], perr_2s[3], popt_2s[2], perr_2s[2], r2_2s)

    # Dopasowanie 2: Model 3-stanowy (poprawny)
    p0_3s = [y_217[0], (y_217[0] + y_217[-1]) / 2.0, y_217[-1], 2.5, 2.2, 2.5, 3.8]
    bounds_3s = ([-np.inf, -np.inf, -np.inf, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    popt_3s, pcov_3s = curve_fit(three_state_model, concs_cd, y_217, p0=p0_3s, bounds=bounds_3s, maxfev=15000)
    perr_3s = np.sqrt(np.diag(pcov_3s))
    y_fit_3s = three_state_model(x_dense, *popt_3s)
    r2_3s = calc_r_squared(y_217, three_state_model(concs_cd, *popt_3s))

    ax_a.plot(
        x_dense, y_fit_3s, color=COLOR_FUSION, ls="-", lw=2.4,
        label=f"3-state fit ($R^2 = {r2_3s:.3f}$)", zorder=3
    )
    # Eksport obu składowych modelu 3-stanowego
    export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm_Step1", TIME_H, CONC_UM, "3-state_trans1",
                   popt_3s[4], perr_3s[4], popt_3s[3], perr_3s[3], r2_3s)
    export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm_Step2", TIME_H, CONC_UM, "3-state_trans2",
                   popt_3s[6], perr_3s[6], popt_3s[5], perr_3s[5], r2_3s)

    ax_a.set_xlim(-0.2, 7.8)
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=13)
    ax_a.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=13)
    ax_a.tick_params(labelsize=11)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower right", bbox_to_anchor=(1, 0.02), fontsize=10.5,
                frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_a.text(0.04, 0.95, "A", transform=ax_a.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_a.set_title("Equilibrium CD Unfolding (217 nm)", fontsize=13, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL B: Widma CD fuzji (210 - 250 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_cd = Normalize(vmin=concs_cd.min(), vmax=concs_cd.max())
    for c in concs_cd:
        wls, mre = cd_data[c]
        mask = (wls >= 210.0) & (wls <= 250.0)
        ax_b.plot(wls[mask], mre[mask], color=plasma_trimmed(norm_cd(c)), lw=1.35)

    ax_b.set_xlim(210.0, 250.0)
    ax_b.set_xlabel("Wavelength (nm)", fontsize=13)
    ax_b.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=13)
    ax_b.tick_params(labelsize=11)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.text(0.04, 0.95, "B", transform=ax_b.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_b.set_title("Far-UV CD Spectra Series", fontsize=13, fontweight="bold", pad=8)

    # Panel B: Prawy dolny róg (y0 = 0.08)
    cax_b = ax_b.inset_axes([0.82, 0.08, 0.035, 0.38])
    sm_b = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd)
    sm_b.set_array([])
    cbar_b = fig.colorbar(sm_b, cax=cax_b, orientation="vertical")
    cbar_b.ax.tick_params(labelsize=9)
    cbar_b.set_label("[GdmCl] (M)", fontsize=9, labelpad=5)

    # --------------------------------------------------------------------------
    # PANEL C: Dual-ratio Fluorescencji (Dwa etapy na podwójnej osi Y)
    # --------------------------------------------------------------------------
    fluo_data = load_fluo_spectra(BASE_DIR_FLUO, CONC_FILE_FLUO)
    concs_fluo = np.array(sorted(fluo_data.keys()))

    # Wyznaczenie stosunków spektralnych
    r1_list, concs_r1 = [], []
    r2_list, concs_r2 = [], []

    for c in concs_fluo:
        w, inten = fluo_data[c]
        i310 = np.interp(310.0, w, inten)
        i330 = np.interp(330.0, w, inten)
        i332 = np.interp(332.0, w, inten)
        i342 = np.interp(342.0, w, inten)

        # Etap 1: TrmD (okno 0 - 4.0 M, I310 / I332)
        if c <= 4.0:
            concs_r1.append(c)
            r1_list.append(i310 / i332)

        # Etap 2: Tm1570 (okno 2.4 - 7.5 M, I342 / I330)
        if c >= 2.4:
            concs_r2.append(c)
            r2_list.append(i342 / i330)

    concs_r1 = np.array(concs_r1)
    r1_vals = np.array(r1_list)
    concs_r2 = np.array(concs_r2)
    r2_vals = np.array(r2_list)

    ax_c1 = ax_c
    ax_c2 = ax_c1.twinx()

    handles_c, labels_c = [], []

    # Lewa oś: Etap 1 (Fioletowe koła)
    sc1 = ax_c1.scatter(concs_r1, r1_vals, color=COLOR_TRMD, marker='o', s=55,
                        edgecolors='none', zorder=4)
    popt_r1, pcov_r1 = curve_fit(two_state_model, concs_r1, r1_vals,
                                 p0=[r1_vals[0], r1_vals[-1], 2.0, 2.2], maxfev=10000)
    perr_r1 = np.sqrt(np.diag(pcov_r1))
    x_dense_r1 = np.linspace(concs_r1.min(), concs_r1.max(), 200)
    line1, = ax_c1.plot(x_dense_r1, two_state_model(x_dense_r1, *popt_r1),
                        color=COLOR_TRMD, lw=2.2, ls='-', zorder=3)
    r2_r1 = calc_r_squared(r1_vals, two_state_model(concs_r1, *popt_r1))
    export_fit_row(FIGURE_ID, CONSTRUCT, "Fluo_DualRatio_Step1_TrmD", TIME_H, CONC_UM, "2-state",
                   popt_r1[3], perr_r1[3], popt_r1[2], perr_r1[2], r2_r1)

    handles_c.append((sc1, line1))
    labels_c.append(r"Step 1: $I_{310}/I_{332}$ (TrmD)")

    # Prawa oś: Etap 2 (Pomarańczowe trójkąty)
    sc2 = ax_c2.scatter(concs_r2, r2_vals, color=COLOR_TM1570, marker='^', s=55,
                        edgecolors='none', zorder=4)
    popt_r2, pcov_r2 = curve_fit(two_state_model, concs_r2, r2_vals,
                                 p0=[r2_vals[0], r2_vals[-1], 2.0, 3.8], maxfev=10000)
    perr_r2 = np.sqrt(np.diag(pcov_r2))
    x_dense_r2 = np.linspace(concs_r2.min(), concs_r2.max(), 200)
    line2, = ax_c2.plot(x_dense_r2, two_state_model(x_dense_r2, *popt_r2),
                        color=COLOR_TM1570, lw=2.2, ls='--', zorder=3)
    r2_r2 = calc_r_squared(r2_vals, two_state_model(concs_r2, *popt_r2))
    export_fit_row(FIGURE_ID, CONSTRUCT, "Fluo_DualRatio_Step2_Tm1570", TIME_H, CONC_UM, "2-state",
                   popt_r2[3], perr_r2[3], popt_r2[2], perr_r2[2], r2_r2)

    handles_c.append((sc2, line2))
    labels_c.append(r"Step 2: $I_{342}/I_{330}$ (Tm1570)")

    ax_c1.set_xlim(-0.2, 7.8)
    ax_c1.set_xlabel("[GdmCl] (M)", fontsize=13)
    ax_c1.set_ylabel(r"Emission Ratio $I_{310} / I_{332}$", fontsize=13, color=COLOR_FUSION)
    ax_c2.set_ylabel(r"Emission Ratio $I_{342} / I_{330}$", fontsize=13, color=COLOR_TM1570)

    ax_c1.tick_params(axis='y', colors=COLOR_FUSION, labelsize=11)
    ax_c2.tick_params(axis='y', colors=COLOR_TM1570, labelsize=11)
    ax_c1.tick_params(axis='x', labelsize=11)
    ax_c1.grid(True, linestyle=":", alpha=0.5)

    # Formatowanie brzegów dla prawej i lewej osi
    ax_c1.spines['left'].set_color(COLOR_FUSION)
    ax_c1.spines['left'].set_linewidth(1.4)
    ax_c2.spines['right'].set_color(COLOR_TM1570)
    ax_c2.spines['right'].set_linewidth(1.4)
    ax_c1.spines['top'].set_visible(False)
    ax_c2.spines['top'].set_visible(False)

    ax_c1.legend(handles_c, labels_c, loc="lower right", bbox_to_anchor=(1, 0.02),
                 fontsize=10.5, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax_c1.text(0.04, 0.95, "C", transform=ax_c1.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_c1.set_title("Dual-Ratio Fluorescence Unfolding", fontsize=13, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL D: Widma emisji fluorescencji fuzji (300 - 400 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_fluo = Normalize(vmin=concs_fluo.min(), vmax=concs_fluo.max())
    for c in concs_fluo:
        w_f, int_f = fluo_data[c]
        mask_f = (w_f >= 300.0) & (w_f <= 400.0)
        ax_d.plot(w_f[mask_f], int_f[mask_f], color=plasma_trimmed(norm_fluo(c)), lw=1.35)

    ax_d.set_xlim(300.0, 400.0)
    ax_d.set_xlabel("Wavelength (nm)", fontsize=13)
    ax_d.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=13)
    ax_d.tick_params(labelsize=11)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.text(0.04, 0.95, "D", transform=ax_d.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_d.set_title("Fluorescence Emission Spectra", fontsize=13, fontweight="bold", pad=8)

    # Panel D: Prawy górny róg (y0 = 0.54)
    cax_d = ax_d.inset_axes([0.82, 0.54, 0.035, 0.38])
    sm_d = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fluo)
    sm_d.set_array([])
    cbar_d = fig.colorbar(sm_d, cax=cax_d, orientation="vertical")
    cbar_d.ax.tick_params(labelsize=9)
    cbar_d.set_label("[GdmCl] (M)", fontsize=9, labelpad=5)

    # Zapis figury w formatach publikacyjnych PNG i PDF
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=300)
    pdf_path = os.path.splitext(OUTPUT_FIGURE_PATH)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf")
    print(f"\n[✓] Gotowa Figura 4.3 została zapisana do:\n  -> {OUTPUT_FIGURE_PATH}\n  -> {pdf_path}")
    print(f"[✓] Zbiorcze parametry dopasowań zaktualizowano w: {CSV_OUTPUT_PATH}")
    # plt.show()


if __name__ == "__main__":
    generate_figure_4_3()