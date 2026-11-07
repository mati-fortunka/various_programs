#!/usr/bin/env python3
"""
plot_refolding_experiment.py

Analiza odwracalności denaturacji (refolding CD) dla trzech konstruktów:
- CnTrmD, CnTm1570 oraz fuzji CnTrmD-Tm1570.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

# ==============================================================================
# 1. Konfiguracja Ścieżek, Danych i Typografii
# ==============================================================================
FIGURE_ID = "Fig_4.5"

BASE_DATA_DIR = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/"
    "equilibrium/biofizyka_CD/refolding/final"
)

OUTPUT_DIR_FIG = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR_FIG, "Figura_4_5_Refolding_CD_2x2_center.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR_FIG, "Figura_4_5_Refolding_CD_2x2_center.png")
CSV_OUTPUT_PATH = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables/thermo_fits_summary.csv"
DETAILED_CSV_PATH = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables/refolding_yield_summary.csv"

HV_CUTOFF = 900.0
PATH_LENGTH_MM = 1.0  # Grubość kuwety w mm
CONC_UM_DEFAULT = 2.0

# Parametry typografii (DejaVu Sans z obsługą bold/italic bez ostrzeżeń)
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

PROTEIN_CONFIGS = [
    {
        "id": "CnTrmD",
        "title": r"$\mathsf{Cn}$TrmD",
        "tag": "A",
        "color": "#1F77B4",  # Morski błękit
        "subdirs": ["trmd", "TrmD", ""],
        "control_candidates": [
            "TrmD_control.csv", "CnTrmD_control.csv", "trmd_control.csv",
            "control.csv", "native.csv", "00001.csv"
        ],
        "refolded_candidates": [
            "TrmD_refolded.csv", "CnTrmD_refolded.csv", "trmd_refolded.csv",
            "refolded.csv", "renatured.csv", "00002.csv"
        ],
        "n_res": 262,
        "conc_uM": 2.6,
    },
    {
        "id": "CnTm1570",
        "title": r"$\mathsf{Cn}$Tm1570",
        "tag": "B",
        "color": "#FF7F0E",  # Ciepły pomarańcz
        "subdirs": ["tm1570", "Tm1570", ""],
        "control_candidates": [
            "Tm1570_control.csv", "CnTm1570_control.csv", "tm1570_control.csv",
            "control.csv", "native.csv", "00001.csv"
        ],
        "refolded_candidates": [
            "Tm1570_refolded.csv", "CnTm1570_refolded.csv", "tm1570_refolded.csv",
            "refolded.csv", "renatured.csv", "00002.csv"
        ],
        "n_res": 224,
        "conc_uM": 4.0,
    },
    {
        "id": "CnTrmD-Tm1570",
        "title": r"Fusion $\mathsf{Cn}$TrmD–Tm1570",
        "tag": "C",
        "color": "#7570B3",  # Fiolet
        "subdirs": ["fuzja", "fusion", "Fusion", ""],
        "control_candidates": [
            "Fusion_control.csv", "CnFusion_control.csv", "fusion_control.csv",
            "control.csv", "native.csv", "00001.csv"
        ],
        "refolded_candidates": [
            "Fusion_refolded.csv", "CnFusion_refolded.csv", "fusion_refolded.csv",
            "refolded.csv", "renatured.csv", "00002.csv"
        ],
        "n_res": 455,
        "conc_uM": 1.7,
    },
]


# ==============================================================================
# 2. Parsowanie ProDataCSV oraz Przetwarzanie Fizykochemiczne
# ==============================================================================
def mdeg_to_mre(mdeg, n_res, conc_uM=CONC_UM_DEFAULT, path_mm=PATH_LENGTH_MM):
    """Przelicza sygnał aparaturowy w mdeg na [theta]_MRW (deg*cm^2*dmol^-1)."""
    return (mdeg * 1e6) / (conc_uM * path_mm * n_res)


def export_refolding_row(fig_id, construct, conc_uM, recovery_217, recovery_222):
    """Dopisuje znormalizowany wiersz wydajności renaturacji do głównej bazy CSV."""
    row = {
        "Figure": fig_id,
        "Construct": construct,
        "Probe": "CD_Refolding_Yield_217nm",
        "Condition_Time": "30h",
        "Condition_Conc_uM": conc_uM,
        "Model": "Refolding_Recovery",
        "Cm_M": None,
        "Cm_SE": None,
        "m_kcal_mol_M": None,
        "m_SE": None,
        "dG_H2O_kcal_mol": round(recovery_217, 1),
        "R_squared": round(recovery_222, 1)
    }
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(CSV_OUTPUT_PATH):
        df_row.to_csv(CSV_OUTPUT_PATH, index=False)
    else:
        df_row.to_csv(CSV_OUTPUT_PATH, mode="a", header=False, index=False)


def parse_prodata_csv(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cd_start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism"):
            cd_start = i + 2
            break

    if cd_start is None:
        return None, None

    cd_lines = []
    for line in lines[cd_start:]:
        l_str = line.strip()
        if not l_str or l_str.startswith("Wavelength") or l_str.startswith("Available"):
            break
        parts = l_str.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                cd_lines.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    if not cd_lines:
        return None, None

    cd_arr = np.array(cd_lines)
    wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

    # Filtracja HV
    hv_start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV"):
            hv_start = i + 2
            break

    if hv_start is not None:
        hv_lines = []
        for line in lines[hv_start:]:
            l_str = line.strip()
            if not l_str or l_str.startswith("Wavelength") or l_str.startswith("Available"):
                break
            parts = l_str.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    hv_lines.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        if hv_lines:
            hv_arr = np.array(hv_lines)
            if len(hv_arr) == len(wls):
                mask = hv_arr[:, 1] <= HV_CUTOFF
                wls, mdeg = wls[mask], mdeg[mask]

    if len(wls) < 10:
        return None, None

    sort_idx = np.argsort(wls)
    return wls[sort_idx], mdeg[sort_idx]


def find_existing_file(candidates, subdirs, base_dir):
    search_dirs = [base_dir]
    for sub in subdirs:
        if sub:
            search_dirs.append(os.path.join(base_dir, sub))

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for cand in candidates:
            fp = os.path.join(d, cand)
            if os.path.exists(fp):
                return fp
        existing_files = os.listdir(d)
        for cand in candidates:
            for ef in existing_files:
                if ef.lower() == cand.lower():
                    return os.path.join(d, ef)
    return None


def process_spectrum(wavelengths, signal_mre, baseline_wl=400.0, window_pts=21, poly_order=2):
    """
    Odejmuje linię bazową oraz wygładza filtrem Savitzky'ego-Golaya.
    Zwiększone okno window_pts=21 i poly_order=2 zapewnia gładki przebieg krzywych.
    """
    if wavelengths.max() >= 350.0:
        near_base = np.where(np.abs(wavelengths - baseline_wl) <= 5.0)[0]
        offset = np.mean(signal_mre[near_base]) if len(near_base) > 0 else signal_mre[-1]
    else:
        far_uv_end = np.where(wavelengths >= (wavelengths.max() - 4.0))[0]
        offset = np.mean(signal_mre[far_uv_end]) if len(far_uv_end) > 0 else 0.0

    signal_corr = signal_mre - offset

    n_pts = len(signal_corr)
    w = window_pts if window_pts % 2 == 1 else window_pts + 1
    if w >= n_pts:
        w = n_pts - 1 if (n_pts - 1) % 2 == 1 else n_pts - 2
    p = min(poly_order, max(1, w - 1))

    if w >= 5:
        signal_smooth = savgol_filter(signal_corr, window_length=w, polyorder=p)
    else:
        signal_smooth = signal_corr

    return signal_corr, signal_smooth, offset


def calculate_recovery_yield(wl_ctrl, sig_ctrl, wl_ref, sig_ref, target_wl=217.0):
    val_c = np.interp(target_wl, wl_ctrl, sig_ctrl)
    val_r = np.interp(target_wl, wl_ref, sig_ref)

    if np.abs(val_c) > 1e-4:
        recovery_pct = (val_r / val_c) * 100.0
    else:
        recovery_pct = np.nan
    return recovery_pct, val_c, val_r


# ==============================================================================
# 3. Główna Procedura Generowania Figury 4.5
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR_FIG, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

    fig = plt.figure(figsize=(13, 11), constrained_layout=True, dpi=300)
    gs = GridSpec(2, 4, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])
    axes = [ax_a, ax_b, ax_c]

    summary_records = []

    print("=" * 80)
    print("ANALIZA ODWRACALNOŚCI ROZWIJANIA CD (REFOLDING W FORMACIE PRODATACSV)")
    print("=" * 80)

    # Gęsta siatka dla interpolacji splajnem, eliminująca kanciastość linii
    dense_wl = np.linspace(210.0, 250.0, 300)

    for idx, cfg in enumerate(PROTEIN_CONFIGS):
        ax = axes[idx]
        prot_id = cfg["id"]
        color_base = cfg["color"]
        n_res = cfg["n_res"]
        conc_uM = cfg["conc_uM"]

        # 1. Lokalizacja plików
        ctrl_path = find_existing_file(cfg["control_candidates"], cfg["subdirs"], BASE_DATA_DIR)
        ref_path = find_existing_file(cfg["refolded_candidates"], cfg["subdirs"], BASE_DATA_DIR)

        if not ctrl_path or not ref_path:
            print(f"[-] [OSTRZEŻENIE] Brak kompletu plików dla {prot_id} w {BASE_DATA_DIR}:")
            print(f"    Kontrola: {ctrl_path}")
            print(f"    Refolded: {ref_path}")
            ax.text(0.5, 0.5, f"Brak danych dla\n{prot_id}", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=13)
            ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=8)
            continue

        # 2. Parsowanie ProDataCSV
        wl_c, mdeg_c = parse_prodata_csv(ctrl_path)
        wl_r, mdeg_r = parse_prodata_csv(ref_path)

        if wl_c is None or wl_r is None:
            print(f"[-] [BŁĄD PARSOWANIA] Niepoprawna struktura ProDataCSV w plikach {prot_id}")
            continue

        # 3. Przeliczenie do [theta]_MRW
        mre_c = mdeg_to_mre(mdeg_c, n_res=n_res, conc_uM=conc_uM)
        mre_r = mdeg_to_mre(mdeg_r, n_res=n_res, conc_uM=conc_uM)

        # 4. Korekta baseline i filtr Savitzky-Golay (ze zwiększonym oknem)
        _, sig_ctrl_smooth, off_c = process_spectrum(wl_c, mre_c, window_pts=21, poly_order=2)
        _, sig_ref_smooth, off_r = process_spectrum(wl_r, mre_r, window_pts=21, poly_order=2)

        # 5. Obliczenie wydajności (% Recovery)
        rec_217, c217, r217 = calculate_recovery_yield(wl_c, sig_ctrl_smooth, wl_r, sig_ref_smooth, 217.0)
        rec_222, c222, r222 = calculate_recovery_yield(wl_c, sig_ctrl_smooth, wl_r, sig_ref_smooth, 222.0)

        print(f"[+] {prot_id}:")
        print(f"    Kontrola : {os.path.basename(ctrl_path)} (baseline offset: {off_c:.1f})")
        print(f"    Refolded : {os.path.basename(ref_path)} (baseline offset: {off_r:.1f})")
        print(f"    Recovery @ 217 nm: {rec_217:.1f}% | @ 222 nm: {rec_222:.1f}%")

        summary_records.append({
            "Construct": prot_id,
            "Residues": n_res,
            "Concentration_uM": conc_uM,
            "Control_217nm_MRW": np.round(c217, 1),
            "Refolded_217nm_MRW": np.round(r217, 1),
            "Recovery_217nm_Percent": np.round(rec_217, 1),
            "Control_222nm_MRW": np.round(c222, 1),
            "Refolded_222nm_MRW": np.round(r222, 1),
            "Recovery_222nm_Percent": np.round(rec_222, 1),
            "Control_File": os.path.basename(ctrl_path),
            "Refolded_File": os.path.basename(ref_path),
        })

        export_refolding_row(FIGURE_ID, prot_id, conc_uM, rec_217, rec_222)

        # 6. Interpolacja splajnem sześciennym dla uzyskania idealnie gładkich linii
        mask_c = (wl_c >= 209.0) & (wl_c <= 251.0)
        mask_r = (wl_r >= 209.0) & (wl_r <= 251.0)

        spl_ctrl = make_interp_spline(wl_c[mask_c], sig_ctrl_smooth[mask_c] / 1000.0, k=3)
        spl_ref = make_interp_spline(wl_r[mask_r], sig_ref_smooth[mask_r] / 1000.0, k=3)

        y_plot_ctrl = spl_ctrl(dense_wl)
        y_plot_ref = spl_ref(dense_wl)

        # Rysowanie wygładzonych widm
        ax.plot(dense_wl, y_plot_ctrl, color=color_base, linestyle="-", lw=2.2, zorder=3)
        ax.plot(dense_wl, y_plot_ref, color=color_base, linestyle="--", dashes=(4.0, 2.5), lw=2.0, zorder=4)

        ax.axhline(0, color="gray", linestyle=":", lw=1.0, alpha=0.7, zorder=1)

        # Skala osi Y i X ustalona wg wytycznych
        ax.set_ylim(-14.5, 0.4)
        ax.set_xlim(210.0, 250.0)

        # Formatowanie osi
        ax.set_title(f"{cfg['title']} – Refolding", fontsize=13, pad=8, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=15)
        ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Etykieta panelu (A, B, C)
        ax.text(0.04, 0.94, cfg["tag"], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="left")

        # Proxy artists dla legendy
        h_ctrl = Line2D([0], [0], color=color_base, linestyle="-", linewidth=2.2)
        h_ref = Line2D([0], [0], color=color_base, linestyle="--", dashes=(4.0, 2.5), linewidth=2.0)

        handles_leg = [h_ctrl, h_ref]
        # labels_leg = [r"Control (native)", f"Refolded (30 h)\n[Recovery: {rec_217:.0f}%]"]
        labels_leg = [r"Control (native)", f"Refolded (30 h)"]

        leg = ax.legend(
            handles_leg, labels_leg,
            loc="lower right", bbox_to_anchor=(0.98, 0.04),
            fontsize=11.5, frameon=True, facecolor="white", framealpha=0.9,
            edgecolor="none", markerfirst=False, handlelength=2.6, handletextpad=0.8
        )
        for text_obj in leg.get_texts():
            text_obj.set_ha("right")

    # 7. Zapis figury w formatach PDF i PNG
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"\n[✓] Gotowa Figura 4.5 zapisana do:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")
    print(f"[✓] Parametry dopasowań zaktualizowano w: {CSV_OUTPUT_PATH}")

    # 8. Zapis szczegółowej tabeli
    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(DETAILED_CSV_PATH, index=False)
        print(f"[✓] Szczegółowe podsumowanie wydajności zapisano do:\n  -> {DETAILED_CSV_PATH}")


if __name__ == "__main__":
    main()