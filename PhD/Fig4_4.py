import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Ścieżek, Parametrów i Palety Projektowej
# ==============================================================================
FIGURE_ID = "Fig_4.4"
CONC_UM = 2.0
PATH_LENGTH_MM = 1.0
TARGET_WL = 217.0
RT = 0.592  # kcal/mol przy 298.15 K
HV_CUTOFF = 900.0

RESIDUES = {
    "TrmD": 262,
    "Tm1570": 224,
    "Fusion": 455
}

TIME_SERIES = [
    {"time_h": 12, "folder_date": "28_07_26", "label": "12 h", "marker": "o", "fill": "full", "ls": "-"},
    {"time_h": 36, "folder_date": "29_07_26", "label": "36 h", "marker": "s", "fill": "full", "ls": "--"},
    {"time_h": 84, "folder_date": "31_07_26", "label": "84 h", "marker": "o", "fill": "none", "ls": ":"}
]

BASE_CD_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD"

OUTPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV"
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "Figura_4_4_Czas_Ekwilibracji_1x3.png")
CSV_OUTPUT_PATH = "tables/thermo_fits_summary.csv"

# Master Palette
COLOR_TRMD = "#1F77B4"    # Marine Blue
COLOR_TM1570 = "#FF7F0E"  # Deep Terracotta
COLOR_FUSION = "#7570B3"  # Muted Purple

INITIAL_GUESSES_3STATE = {
    12: {"m1": 4.83, "d1": 2.19, "m2": 0.77, "d2": 2.78},
    36: {"m1": 3.54, "d1": 2.10, "m2": 0.86, "d2": 3.28},
    84: {"m1": 5.09, "d1": 1.94, "m2": 1.25, "d2": 3.23}
}


# ==============================================================================
# 2. Modele Termodynamiczne i Moduł Eksportu
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Model dwustanowy Santoro-Bolen: N <-> U."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -60, 60))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


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

    print(f"[{fig_id}] {construct} ({time_h}h) | {probe}: Cm={cm_val:.2f} ± {cm_se:.2f} M | m={m_val:.2f} ± {m_se:.2f} | R²={r2_val:.4f}")


# ==============================================================================
# 3. Wczytywanie Danych ProData i Konwersja do [theta]_MRW
# ==============================================================================
def mdeg_to_mre(mdeg, residues, conc_uM=CONC_UM, path_mm=PATH_LENGTH_MM):
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def load_concentrations_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().replace(',', '.')
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine='python')
    df['Sample_number'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    df['den_concentration'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df.dropna(subset=['Sample_number', 'den_concentration'], inplace=True)
    return dict(zip(df['Sample_number'].astype(int), df['den_concentration'].astype(float)))


def load_single_series(construct_subfolder, date_folder, residues):
    folder_path = os.path.join(BASE_CD_DIR, construct_subfolder, "3/2uM", date_folder)
    if not os.path.exists(folder_path):
        alt_date = date_folder.replace("_", ".")
        folder_path = os.path.join(BASE_CD_DIR, construct_subfolder, "3/2uM", alt_date)

    conc_file = os.path.join(folder_path, "concentrations.txt")
    if not os.path.exists(conc_file):
        print(f"⚠️ Brak pliku concentrations w: {folder_path}")
        return None, None

    conc_map = load_concentrations_file(conc_file)
    pattern = re.compile(r"(\d{5})\.csv$")

    concs, mre_vals = [], []

    for fname in sorted(os.listdir(folder_path)):
        m = pattern.search(fname)
        if not m:
            continue
        sn = int(m.group(1))
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
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
                    mask = hv_arr[:, 1] <= HV_CUTOFF
                    wls, mdeg = wls[mask], mdeg[mask]

            if len(wls) < 15:
                continue

            s_idx = np.argsort(wls)
            wls, mdeg = wls[s_idx], mdeg[s_idx]

            if len(mdeg) >= 11:
                mdeg = savgol_filter(mdeg, 11, 3)

            mre = mdeg_to_mre(mdeg, residues)
            concs.append(float(conc))
            mre_vals.append(np.interp(TARGET_WL, wls, mre))
        except Exception:
            continue

    if not concs:
        return None, None

    sort_idx = np.argsort(concs)
    return np.array(concs)[sort_idx], np.array(mre_vals)[sort_idx]


# ==============================================================================
# 4. Generator Figury 4.4 (Układ 1 x 3)
# ==============================================================================
def generate_figure_4_4():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, dpi=300)
    ax_a, ax_b, ax_c = axes[0], axes[1], axes[2]

    x_dense_trmd = np.linspace(0.0, 6.0, 300)
    x_dense_tm = np.linspace(0.0, 7.5, 300)
    x_dense_fus = np.linspace(0.0, 7.5, 300)

    # --------------------------------------------------------------------------
    # PANEL A: CnTrmD (12h, 36h, 84h)
    # --------------------------------------------------------------------------
    handles_a, labels_a = [], []
    for item in TIME_SERIES:
        t_h = item["time_h"]
        c, y = load_single_series("trmd", item["folder_date"], RESIDUES["TrmD"])
        if c is None:
            continue

        facecolor = COLOR_TRMD if item["fill"] == "full" else "none"
        ax_a.scatter(c, y, color=COLOR_TRMD, facecolors=facecolor, marker=item["marker"],
                     s=20, lw=0.9, zorder=3)

        p0 = [y[0], y[-1], 2.0, np.median(c)]
        try:
            popt, pcov = curve_fit(two_state_model, c, y, p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            r2 = calc_r_squared(y, two_state_model(c, *popt))
            ax_a.plot(x_dense_trmd, two_state_model(x_dense_trmd, *popt),
                      color=COLOR_TRMD, ls=item["ls"], lw=1.8, zorder=2)
            export_fit_row(FIGURE_ID, "CnTrmD", "CD_217nm", t_h, CONC_UM, "2-state",
                           popt[3], perr[3], popt[2], perr[2], r2)
        except Exception as e:
            print(f"Fit failed for CnTrmD ({t_h}h): {e}")

        # Połączony uchwyt legendy: linia + punkt
        handle = Line2D(
            [0], [0],
            color=COLOR_TRMD, linestyle=item["ls"], linewidth=1.35,
            marker=item["marker"], markersize=4.5,
            markerfacecolor=facecolor, markeredgecolor=COLOR_TRMD, markeredgewidth=1.0
        )
        handles_a.append(handle)
        labels_a.append(item["label"])

    ax_a.set_xlim(-0.2, 6.2)
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=12)
    ax_a.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_a.tick_params(labelsize=10.5)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(handles_a, labels_a, loc="lower right", bbox_to_anchor=(0.96, 0.05),
                fontsize=10, frameon=True, facecolor="white", framealpha=0.9,
                edgecolor="none", markerfirst=False, handlelength=1.9, handletextpad=0.7)
    ax_a.text(0.04, 0.94, "A", transform=ax_a.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_a.set_title(r"$\boldsymbol{Cn}$" + r"$\bf{TrmD\ (2\ \mu M)}$", fontsize=13, pad=8)

    # --------------------------------------------------------------------------
    # PANEL B: CnTm1570 (12h, 36h, 84h)
    # --------------------------------------------------------------------------
    handles_b, labels_b = [], []
    for item in TIME_SERIES:
        t_h = item["time_h"]
        c, y = load_single_series("tm1570", item["folder_date"], RESIDUES["Tm1570"])
        if c is None:
            continue

        facecolor = COLOR_TM1570 if item["fill"] == "full" else "none"
        ax_b.scatter(c, y, color=COLOR_TM1570, facecolors=facecolor, marker=item["marker"],
                     s=20, lw=0.9, zorder=3)

        p0 = [y[0], y[-1], 2.0, np.median(c)]
        try:
            popt, pcov = curve_fit(two_state_model, c, y, p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            r2 = calc_r_squared(y, two_state_model(c, *popt))
            ax_b.plot(x_dense_tm, two_state_model(x_dense_tm, *popt),
                      color=COLOR_TM1570, ls=item["ls"], lw=1.8, zorder=2)
            export_fit_row(FIGURE_ID, "CnTm1570", "CD_217nm", t_h, CONC_UM, "2-state",
                           popt[3], perr[3], popt[2], perr[2], r2)
        except Exception as e:
            print(f"Fit failed for CnTm1570 ({t_h}h): {e}")

        handle = Line2D(
            [0], [0],
            color=COLOR_TM1570, linestyle=item["ls"], linewidth=1.35,
            marker=item["marker"], markersize=4.5,
            markerfacecolor=facecolor, markeredgecolor=COLOR_TM1570, markeredgewidth=1.0
        )
        handles_b.append(handle)
        labels_b.append(item["label"])

    ax_b.set_xlim(-0.2, 7.8)
    ax_b.set_xlabel("[GdmCl] (M)", fontsize=12)
    ax_b.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_b.tick_params(labelsize=10.5)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(handles_b, labels_b, loc="lower right", bbox_to_anchor=(0.96, 0.05),
                fontsize=10, frameon=True, facecolor="white", framealpha=0.9,
                edgecolor="none", markerfirst=False, handlelength=1.9, handletextpad=0.7)
    ax_b.text(0.04, 0.94, "B", transform=ax_b.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_b.set_title(r"$\boldsymbol{Cn}$" + r"$\bf{Tm1570\ (2\ \mu M)}$", fontsize=13, pad=8)

    # --------------------------------------------------------------------------
    # PANEL C: Fuzja CnTrmD-Tm1570 (12h, 36h, 84h)
    # --------------------------------------------------------------------------
    handles_c, labels_c = [], []
    for item in TIME_SERIES:
        t_h = item["time_h"]
        c, y = load_single_series("fuzja", item["folder_date"], RESIDUES["Fusion"])
        if c is None:
            continue

        facecolor = COLOR_FUSION if item["fill"] == "full" else "none"
        ax_c.scatter(c, y, color=COLOR_FUSION, facecolors=facecolor, marker=item["marker"],
                     s=20, lw=0.9, zorder=3)

        g = INITIAL_GUESSES_3STATE.get(t_h, {"m1": 4.0, "d1": 2.1, "m2": 1.0, "d2": 3.0})
        p0_3s = [
            y[0],
            (y[0] + y[-1]) / 2.0,
            y[-1],
            g["m1"],
            g["d1"],
            g["m2"],
            g["d2"]
        ]
        bounds_3s = (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 2.8, 5.0, 4.5]
        )

        try:
            popt, pcov = curve_fit(three_state_model, c, y, p0=p0_3s, bounds=bounds_3s, maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            r2 = calc_r_squared(y, three_state_model(c, *popt))
            ax_c.plot(x_dense_fus, three_state_model(x_dense_fus, *popt),
                      color=COLOR_FUSION, ls=item["ls"], lw=1.8, zorder=2)
            export_fit_row(FIGURE_ID, "CnTrmD-Tm1570", "CD_217nm_Step1", t_h, CONC_UM, "3-state_trans1",
                           popt[4], perr[4], popt[3], perr[3], r2)
            export_fit_row(FIGURE_ID, "CnTrmD-Tm1570", "CD_217nm_Step2", t_h, CONC_UM, "3-state_trans2",
                           popt[6], perr[6], popt[5], perr[5], r2)
        except Exception as e:
            print(f"3-state fit failed for Fusion ({t_h}h): {e}. Fallback to 2-state.")
            popt_2s, pcov_2s = curve_fit(two_state_model, c, y, p0=[y[0], y[-1], 2.0, 2.5], maxfev=10000)
            perr_2s = np.sqrt(np.diag(pcov_2s))
            r2_2s = calc_r_squared(y, two_state_model(c, *popt_2s))
            ax_c.plot(x_dense_fus, two_state_model(x_dense_fus, *popt_2s),
                      color=COLOR_FUSION, ls=item["ls"], lw=1.6, zorder=2)
            export_fit_row(FIGURE_ID, "CnTrmD-Tm1570", "CD_217nm_Fallback", t_h, CONC_UM, "2-state",
                           popt_2s[3], perr_2s[3], popt_2s[2], perr_2s[2], r2_2s)

        handle = Line2D(
            [0], [0],
            color=COLOR_FUSION, linestyle=item["ls"], linewidth=1.35,
            marker=item["marker"], markersize=4.5,
            markerfacecolor=facecolor, markeredgecolor=COLOR_FUSION, markeredgewidth=1.0
        )
        handles_c.append(handle)
        labels_c.append(item["label"])

    ax_c.set_xlim(-0.2, 7.8)
    ax_c.set_xlabel("[GdmCl] (M)", fontsize=12)
    ax_c.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=12)
    ax_c.tick_params(labelsize=10.5)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(handles_c, labels_c, loc="lower right", bbox_to_anchor=(0.96, 0.05),
                fontsize=10, frameon=True, facecolor="white", framealpha=0.9,
                edgecolor="none", markerfirst=False, handlelength=1.9, handletextpad=0.7)
    ax_c.text(0.04, 0.94, "C", transform=ax_c.transAxes, fontsize=15, fontweight="bold", va="top")
    ax_c.set_title(r"$\bf{Fusion\ }$" + r"$\boldsymbol{Cn}$" + r"$\bf{TrmD–Tm1570\ (2\ \mu M)}$", fontsize=13, pad=8)

    # Zapis w wysokiej rozdzielczości PNG oraz wektorowym PDF
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=300)
    pdf_path = os.path.splitext(OUTPUT_FIGURE_PATH)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf")

    print(f"\n[✓] Zaktualizowana Figura 4.4 zapisana do:\n  -> {OUTPUT_FIGURE_PATH}\n  -> {pdf_path}")
    print(f"[✓] Zbiorcze parametry dopasowań zaktualizowano w: {CSV_OUTPUT_PATH}")
    # plt.show()


if __name__ == "__main__":
    generate_figure_4_4()