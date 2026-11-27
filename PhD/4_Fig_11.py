import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Czcionek, Ścieżek i Palety Barw (Spójna z Rozdziałem 4)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # Pogrubiona kursywa (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

RT = 0.592  # kcal/mol przy 298.15 K
PATH_LENGTH_MM = 1.0

# Ścieżki danych CD (format ProData) — zgodne z 4_Fig_3.py, 4_Fig_5.py, 4_Fig_9.py
DIR_CD_CN_TRMD = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/"
    "equilibrium/biofizyka_CD/trmd/3/2uM/28_07_26"
)
DIR_CD_CN_TM1570 = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/"
    "equilibrium/biofizyka_CD/tm1570/3/2uM/28_07_26"
)
DIR_CD_CN_FUSION = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/"
    "equilibrium/biofizyka_CD/fuzja/3/2uM/28_07_26"
)
DIR_CD_FS_FUSION = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/"
    "fusions/F8E4N/equilibrium/unfolding/2nd_set/CD/fuzja_F8_unfolding_22h"
)

# Ścieżki wyjściowe
OUTPUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Figura_4_11_Summary_Comparison_1x2.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figura_4_11_Summary_Comparison_1x2.png")

# Konfiguracja konstruktów (parametry fizykochemiczne i preprocessing)
CONSTRUCT_CFG = {
    "CnTrmD": {
        "dir": DIR_CD_CN_TRMD,
        "residues": 262,
        "conc_uM": 2.0,
        "hv_cutoff": 875.0,
        "savgol_window": 11,
        "file_pattern": r"(\d{5})\.csv$",
    },
    "CnTm1570": {
        "dir": DIR_CD_CN_TM1570,
        "residues": 224,
        "conc_uM": 2.0,
        "hv_cutoff": 875.0,
        "savgol_window": 11,
        "file_pattern": r"(\d{5})\.csv$",
    },
    "CnFusion": {
        "dir": DIR_CD_CN_FUSION,
        "residues": 455,
        "conc_uM": 2.0,
        "hv_cutoff": 850.0,
        "savgol_window": 11,
        "file_pattern": r"(\d{5})\.csv$",
    },
    "FsFusion": {
        "dir": DIR_CD_FS_FUSION,
        "residues": 447,
        "conc_uM": 1.0,
        "hv_cutoff": 850.0,
        "savgol_window": 15,
        "file_pattern": r"(\d+)\.csv$",
    },
}

# Master Palette projektu
COLOR_TRMD = "#1F77B4"      # Classic Marine Blue (Fig. 4.3)
COLOR_TM1570 = "#FF7F0E"    # Deep Terracotta / Orange (Fig. 4.3)
COLOR_CN_FUS = "#7570B3"    # Purple (Fig. 4.5)
COLOR_FS_PTS = "#17BECF"    # Teal / Cyan — punkty Fs (Fig. 4.9)
COLOR_FS_LINE = "#0E717B"   # Dark Teal — dopasowanie Fs (Fig. 4.9)


# ==============================================================================
# 2. Modele Termodynamiczne i Normalizacja
# ==============================================================================
def two_state_dimer_model(x, a_n, a_d, m, d, conc_uM=2.0):
    """
    Model dwustanowy rozwijania z dysocjacją dla homodimeru: N2 <-> 2D.
    Stosowany dla izolowanych domen CnTrmD oraz CnTm1570.
    """
    P_t = conc_uM * 1e-6
    K_u = P_t * np.exp(np.clip((m * (x - d)) / RT, -500.0, 500.0))
    f_d = 2.0 / (1.0 + np.sqrt(1.0 + (8.0 * P_t) / K_u))
    f_n2 = 1.0 - f_d
    return a_n * f_n2 + a_d * f_d


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """
    Model trójstanowy dla monomeru: N <-> I <-> U.
    Stosowany dla obu fuzji (CnTrmD-Tm1570 oraz FsTrmD-Tm1570).
    """
    sig1 = 1.0 / (1.0 + np.exp(np.clip(-(m1 * (x - d1)) / RT, -60.0, 60.0)))
    sig2 = 1.0 / (1.0 + np.exp(np.clip(-(m2 * (x - d2)) / RT, -60.0, 60.0)))
    return a_n * (1.0 - sig1) + a_i * (sig1 - sig2) + a_u * sig2


def normalize_curve(y, y_start, y_end):
    """
    Przekształca liniowo sygnał (punkty i dopasowaną krzywą tą samą transformacją),
    tak aby wartość początkowa krzywej (y_start) wynosiła 0.0, a końcowa (y_end) 1.0.
    """
    span = y_end - y_start
    if np.isclose(span, 0.0):
        raise ValueError("Zerowa amplituda krzywej dopasowania — nie można znormalizować.")
    return (y - y_start) / span


def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# ==============================================================================
# 3. Wczytywanie Danych CD (Format ProData CSV)
# ==============================================================================
def mdeg_to_mre(mdeg, residues, conc_uM, path_mm=PATH_LENGTH_MM):
    """Przelicza mdeg na [theta]_MRW [deg * cm^2 * dmol^-1]."""
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def load_concentrations(folder_path):
    """Wczytuje mapowanie Sample_number -> den_concentration z concentrations.txt."""
    candidates = [
        os.path.join(folder_path, "concentrations.txt"),
        os.path.join(os.path.dirname(folder_path), "concentrations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(folder_path)), "concentrations.txt"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Nie znaleziono concentrations.txt dla {folder_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace(",", ".")
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
    df["Sample_number"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["den_concentration"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
    return dict(zip(df["Sample_number"].astype(int), df["den_concentration"].astype(float)))


def load_cd_217(cfg):
    """Wczytuje widma CD ProData dla zadanego konstruktu i zwraca (concs, mre_217)."""
    folder_path = cfg["dir"]
    conc_map = load_concentrations(folder_path)
    pattern = re.compile(cfg["file_pattern"], re.IGNORECASE)
    spectra_by_conc = {}

    for fname in sorted(os.listdir(folder_path)):
        m = pattern.search(fname)
        if not m:
            continue
        sn = int(m.group(1))
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        cd_idx = next(
            (
                i
                for i in range(len(lines) - 1)
                if lines[i].strip().startswith("Wavelength")
                and lines[i + 1].strip().startswith("CircularDichroism")
            ),
            None,
        )
        if cd_idx is None:
            continue

        cd_lines = []
        for ln in lines[cd_idx + 2:]:
            s = ln.strip()
            if not s or s.startswith("Wavelength") or s.startswith("Available"):
                break
            cd_lines.append(list(map(float, ln.replace(",", " ").split())))

        cd_arr = np.array(cd_lines)
        wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

        # Filtracja HV
        hv_idx = next(
            (
                i
                for i in range(len(lines) - 1)
                if lines[i].strip().startswith("Wavelength")
                and lines[i + 1].strip().startswith("HV")
            ),
            None,
        )
        if hv_idx is not None:
            hv_lines = []
            for ln in lines[hv_idx + 2:]:
                s = ln.strip()
                if not s or s.startswith("Wavelength") or s.startswith("Available"):
                    break
                hv_lines.append(list(map(float, ln.replace(",", " ").split())))
            hv_arr = np.array(hv_lines)
            if len(hv_arr) == len(wls):
                mask = hv_arr[:, 1] <= cfg["hv_cutoff"]
                wls, mdeg = wls[mask], mdeg[mask]

        if len(wls) < 15:
            continue

        s_idx = np.argsort(wls)
        wls, mdeg = wls[s_idx], mdeg[s_idx]

        win = cfg["savgol_window"]
        if len(mdeg) >= win:
            mdeg = savgol_filter(mdeg, win, 3)

        mre = mdeg_to_mre(mdeg, cfg["residues"], cfg["conc_uM"])
        spectra_by_conc[float(conc)] = (wls, mre)

    concs = np.array(sorted(spectra_by_conc.keys()))
    y_217 = np.array(
        [np.interp(217.0, spectra_by_conc[c][0], spectra_by_conc[c][1]) for c in concs]
    )
    return concs, y_217


# ==============================================================================
# 4. Dopasowania i Generator Figury Zbiorczej (1 x 2)
# ==============================================================================
def fit_domain_2state_dimer(concs, y_raw, conc_uM):
    fit_fn = lambda x, a_n, a_d, m, d: two_state_dimer_model(x, a_n, a_d, m, d, conc_uM)
    p0 = [y_raw[0], y_raw[-1], 2.0, np.median(concs)]
    bounds = ([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(fit_fn, concs, y_raw, p0=p0, bounds=bounds, maxfev=15000)
    perr = np.sqrt(np.diag(pcov))

    y_fit_at_concs = fit_fn(concs, *popt)
    r2_raw = calc_r_squared(y_raw, y_fit_at_concs)

    x_dense = np.linspace(concs.min(), concs.max(), 350)
    y_fit_dense = fit_fn(x_dense, *popt)

    # Normalizacja względem krańców dopasowanej krzywej: start (x_min) -> 0, koniec (x_max) -> 1
    y_start, y_end = y_fit_dense[0], y_fit_dense[-1]
    y_norm = normalize_curve(y_raw, y_start, y_end)
    y_fit_norm = normalize_curve(y_fit_dense, y_start, y_end)
    r2_norm = calc_r_squared(y_norm, normalize_curve(y_fit_at_concs, y_start, y_end))

    return x_dense, y_norm, y_fit_norm, popt, perr, r2_raw, r2_norm


def fit_fusion_3state_monomer(concs, y_raw, p0_steps):
    p0 = [y_raw[0], (y_raw[0] + y_raw[-1]) / 2.0, y_raw[-1], *p0_steps]
    bounds = (
        [-np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    )
    popt, pcov = curve_fit(three_state_model, concs, y_raw, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    y_fit_at_concs = three_state_model(concs, *popt)
    r2_raw = calc_r_squared(y_raw, y_fit_at_concs)

    x_dense = np.linspace(concs.min(), concs.max(), 350)
    y_fit_dense = three_state_model(x_dense, *popt)

    # Normalizacja względem krańców dopasowanej krzywej: start (x_min) -> 0, koniec (x_max) -> 1
    y_start, y_end = y_fit_dense[0], y_fit_dense[-1]
    y_norm = normalize_curve(y_raw, y_start, y_end)
    y_fit_norm = normalize_curve(y_fit_dense, y_start, y_end)
    r2_norm = calc_r_squared(y_norm, normalize_curve(y_fit_at_concs, y_start, y_end))

    return x_dense, y_norm, y_fit_norm, popt, perr, r2_raw, r2_norm


def generate_summary_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Wczytanie krzywych CD (217 nm)
    c_trmd, y_trmd = load_cd_217(CONSTRUCT_CFG["CnTrmD"])
    c_tm, y_tm = load_cd_217(CONSTRUCT_CFG["CnTm1570"])
    c_cn_fus, y_cn_fus = load_cd_217(CONSTRUCT_CFG["CnFusion"])
    c_fs_fus, y_fs_fus = load_cd_217(CONSTRUCT_CFG["FsFusion"])

    # 2. Dopasowania termodynamiczne (domeny: 2-state dimer; fuzje: 3-state monomer)
    xd_trmd, yn_trmd, yf_trmd, p_trmd, e_trmd, r2_trmd, r2n_trmd = fit_domain_2state_dimer(
        c_trmd, y_trmd, CONSTRUCT_CFG["CnTrmD"]["conc_uM"]
    )
    xd_tm, yn_tm, yf_tm, p_tm, e_tm, r2_tm, r2n_tm = fit_domain_2state_dimer(
        c_tm, y_tm, CONSTRUCT_CFG["CnTm1570"]["conc_uM"]
    )
    xd_cn, yn_cn, yf_cn, p_cn, e_cn, r2_cn, r2n_cn = fit_fusion_3state_monomer(
        c_cn_fus, y_cn_fus, p0_steps=(5.0, 2.2, 1.0, 2.8)
    )
    xd_fs, yn_fs, yf_fs, p_fs, e_fs, r2_fs, r2n_fs = fit_fusion_3state_monomer(
        c_fs_fus, y_fs_fus, p0_steps=(3.1, 2.3, 0.9, 3.7)
    )

    # Podsumowanie kontrolne w konsoli (parametry + weryfikacja [0, 1] i spójności punktów)
    print("=== Weryfikacja dopasowań i normalizacji (CD 217 nm) ===")
    print(
        f"CnTrmD        (2-state dimer):   Cm={p_trmd[3]:.2f}±{e_trmd[3]:.2f} M, m={p_trmd[2]:.2f}±{e_trmd[2]:.2f}, "
        f"R²={r2_trmd:.4f} (norm={r2n_trmd:.4f}), fit=[{yf_trmd[0]:.3f}, {yf_trmd[-1]:.3f}]"
    )
    print(
        f"CnTm1570      (2-state dimer):   Cm={p_tm[3]:.2f}±{e_tm[3]:.2f} M, m={p_tm[2]:.2f}±{e_tm[2]:.2f}, "
        f"R²={r2_tm:.4f} (norm={r2n_tm:.4f}), fit=[{yf_tm[0]:.3f}, {yf_tm[-1]:.3f}]"
    )
    print(
        f"CnTrmD-Tm1570 (3-state monomer): Cm1={p_cn[4]:.2f}±{e_cn[4]:.2f} M (m1={p_cn[3]:.2f}), "
        f"Cm2={p_cn[6]:.2f}±{e_cn[6]:.2f} M (m2={p_cn[5]:.2f}), "
        f"R²={r2_cn:.4f} (norm={r2n_cn:.4f}), fit=[{yf_cn[0]:.3f}, {yf_cn[-1]:.3f}]"
    )
    print(
        f"FsTrmD-Tm1570 (3-state monomer): Cm1={p_fs[4]:.2f}±{e_fs[4]:.2f} M (m1={p_fs[3]:.2f}), "
        f"Cm2={p_fs[6]:.2f}±{e_fs[6]:.2f} M (m2={p_fs[5]:.2f}), "
        f"R²={r2_fs:.4f} (norm={r2n_fs:.4f}), fit=[{yf_fs[0]:.3f}, {yf_fs[-1]:.3f}]"
    )

    # Automatyczne zabezpieczenie zakresu osi y, aby żaden punkt nie został ucięty
    all_yn = np.concatenate([yn_trmd, yn_tm, yn_cn, yn_fs])
    ylim_low = min(-0.08, float(np.min(all_yn)) - 0.04)
    ylim_high = max(1.08, float(np.max(all_yn)) + 0.04)

    # 3. Rysowanie figury 1 x 2 ze znormalizowaną osią y
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.2, 5.2), sharey=True, constrained_layout=True, dpi=300
    )

    # --------------------------------------------------------------------------
    # PANEL A: CnTrmD and CnTm1570 vs CnTrmD-Tm1570
    # --------------------------------------------------------------------------
    ax_a.plot(xd_trmd, yf_trmd, color=COLOR_TRMD, lw=2.2, zorder=2)
    ax_a.plot(xd_tm, yf_tm, color=COLOR_TM1570, lw=2.2, zorder=2)
    ax_a.plot(xd_cn, yf_cn, color=COLOR_CN_FUS, lw=2.4, zorder=3)

    ax_a.scatter(c_trmd, yn_trmd, color=COLOR_TRMD, marker="o", s=52, edgecolors="none", zorder=4)
    ax_a.scatter(c_tm, yn_tm, color=COLOR_TM1570, marker="s", s=48, edgecolors="none", zorder=4)
    ax_a.scatter(c_cn_fus, yn_cn, color=COLOR_CN_FUS, marker="^", s=58, edgecolors="none", zorder=5)

    handles_a = [
        Line2D([0], [0], color=COLOR_TRMD, marker="o", markersize=7.0, markeredgecolor="none", lw=2.2),
        Line2D([0], [0], color=COLOR_TM1570, marker="s", markersize=6.6, markeredgecolor="none", lw=2.2),
        Line2D([0], [0], color=COLOR_CN_FUS, marker="^", markersize=7.2, markeredgecolor="none", lw=2.4),
    ]
    labels_a = [
        r"$\mathit{Cn}$TrmD",
        r"$\mathit{Cn}$Tm1570",
        r"$\mathit{Cn}$TrmD–Tm1570",
    ]

    ax_a.set_xlim(-0.2, 7.8)
    ax_a.set_ylim(ylim_low, ylim_high)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_a.set_ylabel(r"Normalized CD Signal at 217 nm ($f_{\mathrm{U}}$)", fontsize=15)
    ax_a.tick_params(labelsize=13)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    leg_a = ax_a.legend(
        handles_a, labels_a, loc="lower right", bbox_to_anchor=(0.98, 0.03),
        fontsize=11.5, frameon=True, facecolor="white", framealpha=0.92,
        edgecolor="none", markerfirst=False, handlelength=2.4, handletextpad=0.8
    )
    for text_obj in leg_a.get_texts():
        text_obj.set_ha("right")

    ax_a.text(0.04, 0.95, "A", transform=ax_a.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_a.set_title(
        r"$\mathsf{Cn}$TrmD and $\mathsf{Cn}$Tm1570 vs $\mathsf{Cn}$TrmD–Tm1570",
        fontsize=12.5, fontweight="bold", pad=8
    )

    # --------------------------------------------------------------------------
    # PANEL B: CnTrmD-Tm1570 vs FsTrmD-Tm1570
    # --------------------------------------------------------------------------
    ax_b.plot(xd_cn, yf_cn, color=COLOR_CN_FUS, lw=2.4, zorder=2)
    ax_b.plot(xd_fs, yf_fs, color=COLOR_FS_LINE, lw=2.4, zorder=3)

    ax_b.scatter(c_cn_fus, yn_cn, color=COLOR_CN_FUS, marker="^", s=58, edgecolors="none", zorder=4)
    ax_b.scatter(c_fs_fus, yn_fs, color=COLOR_FS_PTS, marker="D", s=52, edgecolors="none", zorder=5)

    handles_b = [
        Line2D([0], [0], color=COLOR_CN_FUS, marker="^", markersize=7.2, markeredgecolor="none", lw=2.4),
        Line2D(
            [0], [0], color=COLOR_FS_LINE, marker="D", markersize=6.6,
            markerfacecolor=COLOR_FS_PTS, markeredgecolor="none", lw=2.4
        ),
    ]
    labels_b = [
        r"$\mathit{Cn}$TrmD–Tm1570",
        r"$\mathit{Fs}$TrmD–Tm1570",
    ]

    ax_b.set_xlim(-0.2, 7.8)
    ax_b.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_b.tick_params(axis="both", labelsize=13, labelleft=True)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    leg_b = ax_b.legend(
        handles_b, labels_b, loc="lower right", bbox_to_anchor=(0.98, 0.03),
        fontsize=11.5, frameon=True, facecolor="white", framealpha=0.92,
        edgecolor="none", markerfirst=False, handlelength=2.4, handletextpad=0.8
    )
    for text_obj in leg_b.get_texts():
        text_obj.set_ha("right")

    ax_b.text(0.04, 0.95, "B", transform=ax_b.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_b.set_title(
        r"$\mathsf{Cn}$TrmD–Tm1570 vs $\mathsf{Fs}$TrmD–Tm1570",
        fontsize=12.5, fontweight="bold", pad=8
    )

    # 4. Zapis w formatach publikacyjnych PDF i PNG
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"\n[✓] Figura zbiorcza zapisana do:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")


if __name__ == "__main__":
    generate_summary_figure()