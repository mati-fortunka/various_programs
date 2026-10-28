#!/usr/bin/env python3
"""
plot_refolding_experiment.py

Analiza odwracalności denaturacji (refolding CD) dla trzech konstruktów:
- TrmD, Tm1570 oraz fuzji CnTrmD-Tm1570.

Procedura:
1. Parsowanie formatu ProDataCSV z ekstrakcją bloków CircularDichroism i HV.
2. Filtracja fotopowielacza (HV <= 900 V).
3. Przeliczenie eliptyczności mdeg na [theta]_MRW (deg*cm^2*dmol^-1).
4. Korekta linii bazowej i wygładzanie Savitzky'ego-Golaya (w=15, p=3).
5. Wyznaczenie wydajności renaturacji (% Recovery) przy 217 nm i 222 nm.
6. Generowanie publikacyjnej ryciny 1x3 (PDF i PNG 300 DPI) oraz tabeli CSV.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja Ścieżek, Danych i Typografii
# ==============================================================================
BASE_DATA_DIR = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/"
    "equilibrium/biofizyka_CD/refolding/final"
)

OUTPUT_DIR_FIG = "figures/ch04"
OUTPUT_DIR_TAB = "tables"
OUTPUT_PDF = os.path.join(OUTPUT_DIR_FIG, "Fig_4_6_Refolding_CD_1x3.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR_FIG, "Fig_4_6_Refolding_CD_1x3.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR_TAB, "refolding_yield_summary.csv")

HV_CUTOFF = 900.0
PATH_LENGTH_MM = 1.0  # Grubość kuwety w mm
CONC_UM_DEFAULT = 2.0  # Domyślne stężenie pomiarowe po rozcieńczeniu [uM]

# Parametry typografii (DejaVu Sans z obsługą bold/italic bez ostrzeżeń)
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"

PROTEIN_CONFIGS = [
    {
        "id": "TrmD",
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
        "id": "Tm1570",
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
        "conc_uM": 4,
    },
    {
        "id": "Fusion",
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


def parse_prodata_csv(filepath):
    """
    Parsuje pliki ProDataCSV:
    - ekstrahuje bloki CircularDichroism oraz HV,
    - usuwa punkty z nasyceniem detektora (HV > HV_CUTOFF).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1. Wyszukanie bloku CircularDichroism
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

    # 2. Wyszukanie bloku HV i filtracja nasycenia
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

    # Sortowanie według rosnącej długości fali
    sort_idx = np.argsort(wls)
    return wls[sort_idx], mdeg[sort_idx]


def find_existing_file(candidates, subdirs, base_dir):
    """Wyszukuje plik spośród listy wzorców w katalogu głównym lub podkatalogach."""
    search_dirs = [base_dir]
    for sub in subdirs:
        if sub:
            search_dirs.append(os.path.join(base_dir, sub))

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # Bezpośrednie dopasowanie nazwy
        for cand in candidates:
            fp = os.path.join(d, cand)
            if os.path.exists(fp):
                return fp
        # Dopasowanie niewrażliwe na wielkość liter
        existing_files = os.listdir(d)
        for cand in candidates:
            for ef in existing_files:
                if ef.lower() == cand.lower():
                    return os.path.join(d, ef)
    return None


def process_spectrum(wavelengths, signal_mre, baseline_wl=400.0, window_pts=15, poly_order=3):
    """
    Odejmuje linię bazową (przy 400 nm lub przy skrajnej długości fali Far-UV)
    oraz wygładza filtrem Savitzky'ego-Golaya.
    """
    # Wybór punktu bazowego: jeśli widmo sięga 400 nm, bierzemy okolice 400 nm;
    # w przeciwnym razie bierzemy pasmo skrajne (250-260 nm), gdzie struktura drugorzędowa nie daje sygnału
    if wavelengths.max() >= 350.0:
        near_base = np.where(np.abs(wavelengths - baseline_wl) <= 5.0)[0]
        offset = np.mean(signal_mre[near_base]) if len(near_base) > 0 else signal_mre[-1]
    else:
        far_uv_end = np.where(wavelengths >= (wavelengths.max() - 4.0))[0]
        offset = np.mean(signal_mre[far_uv_end]) if len(far_uv_end) > 0 else 0.0

    signal_corr = signal_mre - offset

    # Wygładzanie Savitzky'ego-Golaya
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
    """Oblicza procentowy stopień odzysku sygnału eliptyczności przy danej długości fali."""
    val_c = np.interp(target_wl, wl_ctrl, sig_ctrl)
    val_r = np.interp(target_wl, wl_ref, sig_ref)

    if np.abs(val_c) > 1e-4:
        recovery_pct = (val_r / val_c) * 100.0
    else:
        recovery_pct = np.nan
    return recovery_pct, val_c, val_r


# ==============================================================================
# 3. Główna Procedura Generowania Figury 4.9
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR_FIG, exist_ok=True)
    os.makedirs(OUTPUT_DIR_TAB, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300, constrained_layout=True)
    summary_records = []

    print("=" * 80)
    print("ANALIZA ODWRACALNOŚCI ROZWIJANIA CD (REFOLDING W FORMACIE PRODATACSV)")
    print("=" * 80)

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
                    transform=ax.transAxes, color="gray", fontsize=11)
            ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=8)
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

        # 4. Korekta baseline i filtr Savitzky-Golay
        _, sig_ctrl_smooth, off_c = process_spectrum(wl_c, mre_c)
        _, sig_ref_smooth, off_r = process_spectrum(wl_r, mre_r)

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

        # 6. Wizualizacja spektralna
        # Widmo natywne (linia ciągła)
        ax.plot(wl_c, sig_ctrl_smooth, color=color_base, linestyle="-", lw=2.2,
                label=r"Control (native)", zorder=3)

        # Widmo po renaturacji (linia przerywana z wyliczoną wydajnością)
        ax.plot(wl_r, sig_ref_smooth, color=color_base, linestyle="--", lw=2.0,
                label=f"Refolded (30 h)", zorder=4)

        # Pozioma linia zerowa
        ax.axhline(0, color="gray", linestyle=":", lw=0.9, alpha=0.7, zorder=1)

        # Formatowanie osi
        ax.set_title(f"{cfg['title']} – Refolding", fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=11)

        if idx == 0:
            ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ (\mathrm{deg}\cdot\mathrm{cm}^2\cdot\mathrm{dmol}^{-1})$",
                          fontsize=11)

        # Ograniczenie zakresu do czytelnego okna Far-UV
        min_wl = max(205.0, min(wl_c.min(), wl_r.min()))
        max_wl = min(260.0, max(wl_c.max(), wl_r.max()))
        ax.set_xlim(min_wl, max_wl)

        # Etykieta panelu (A, B, C)
        ax.text(0.03, 0.96, cfg["tag"], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.90))

        ax.legend(frameon=True, fontsize=9.5, loc="lower right", facecolor="white", edgecolor="none")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=10)

    # 7. Zapis figury
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"\n[✓] Zapisano wykresy do:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")

    # 8. Zapis tabeli podsumowującej
    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(OUTPUT_CSV, index=False)
        print(f"[✓] Zapisano podsumowanie wydajności do:\n  -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()