import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Matplotlib i Parametrów Doświadczenia
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # POGRUBIONA KURSYWA (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"             # Wycisza ostrzeżenie o 'cursive'
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

FIGURE_ID = "Fig_4.7"
CONSTRUCT = "FsTrmD-Tm1570"
TIME_H = 22
CONC_UM = 1.0       # Stężenie białka [uM]
PATH_MM = 1.0       # Droga optyczna kuwety [mm]
N_RES = 447         # Liczba reszt dla FsTrmD-Tm1570 (UniProt: F8E4N)
RT = 0.592          # kcal/mol przy 298.15 K
HV_CUTOFF = 950.0

# Zakres CSM/AEW dla fluorescencji
CSM_MIN_WL = 310.0
CSM_MAX_WL = 400.0

# Master Palette
COLOR_FS = "#17BECF"        # Teal / Stalowy cyjan
COLOR_FS_DARK = "#0E717B"   # Ciemny akcent / linia 3-state
COLOR_2STATE = "#7F7F7F"    # Szary dla fitu 2-state
COLOR_RATIO = "#7570B3"     # Fiolet dla panelu ratiometrycznego

COLOR_ALPHA = "#D62728"     # K2D Alpha (czerwony)
COLOR_BETA = "#1F77B4"      # K2D Beta (niebieski)
COLOR_COIL = "#FF7F0E"      # K2D Coil (pomarańczowy)

# Gradient plasma
plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)

# Ścieżki danych
CD_DATA_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/CD/fuzja_F8_unfolding_22h"
FLUO_DATA_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/fluo/spectra_F8_22h_26_04"

OUTPUT_DIR = "figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Figura_4_7_FsTrmD_Tm1570_summary_2x3.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figura_4_7_FsTrmD_Tm1570_summary_2x3.png")
CSV_PATH = "tables/thermo_fits_summary.csv"


# ==============================================================================
# 2. Równania Modeli Termodynamicznych i Dekonwolucja K2D
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Dwustanowy model równowagowy Santoro-Bolen."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -60, 60))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """Trójstanowy model równowagowy: N <-> I <-> U."""
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
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(CSV_PATH):
        df_row.to_csv(CSV_PATH, index=False)
    else:
        df_row.to_csv(CSV_PATH, mode="a", header=False, index=False)

    print(f"[{fig_id}] {construct} | {probe} ({model_name}): Cm={cm_val:.3f} ± {cm_se:.3f} M | m={m_val:.3f} ± {m_se:.3f} | R²={r2_val:.4f}")


def estimate_k2d_secondary_structure(wls, mre):
    """
    Wyznacza frakcje struktury drugorzędowej (Alpha, Beta, Coil)
    z użyciem formalizmu K2D na bazie sygnału delta_epsilon przy 208 i 222 nm.
    """
    delta_eps_208 = np.interp(208.0, wls, mre) / 3298.0
    delta_eps_222 = np.interp(222.0, wls, mre) / 3298.0

    alpha = np.clip((-delta_eps_222 - 0.5) / 9.5, 0.0, 1.0)
    beta = np.clip((-0.45 * delta_eps_208 + 0.30 * delta_eps_222 + 1.2) / 6.0, 0.0, 1.0 - alpha)
    coil = np.clip(1.0 - (alpha + beta), 0.0, 1.0)

    total = alpha + beta + coil
    if total > 0:
        return alpha / total, beta / total, coil / total
    return 0.0, 0.3, 0.7


# ==============================================================================
# 3. Parsowanie concentrations.txt oraz Wczytywanie Danych CD i Fluo
# ==============================================================================
def mdeg_to_mre(mdeg, conc_uM=CONC_UM, path_mm=PATH_MM, residues=N_RES):
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def load_concentrations(folder_path):
    candidates = [
        os.path.join(folder_path, "concentrations.txt"),
        os.path.join(os.path.dirname(folder_path), "concentrations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(folder_path)), "concentrations.txt")
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


def load_cd_prodata(folder_path):
    conc_map = load_concentrations(folder_path)
    pattern = re.compile(r"(\d+)\.csv$", re.IGNORECASE)
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

        cd_idx = next((i for i in range(len(lines) - 1)
                       if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism")), None)
        if cd_idx is None:
            continue

        cd_lines = []
        for ln in lines[cd_idx + 2:]:
            if not ln.strip() or ln.strip().startswith("Wavelength") or ln.strip().startswith("Available"):
                break
            cd_lines.append(list(map(float, ln.replace(",", " ").split())))

        cd_arr = np.array(cd_lines)
        wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

        # Filtracja HV
        hv_idx = next((i for i in range(len(lines) - 1)
                       if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")), None)
        if hv_idx is not None:
            hv_lines = []
            for ln in lines[hv_idx + 2:]:
                if not ln.strip() or ln.strip().startswith("Wavelength") or ln.strip().startswith("Available"):
                    break
                hv_lines.append(list(map(float, ln.replace(",", " ").split())))
            hv_arr = np.array(hv_lines)
            if len(hv_arr) == len(wls):
                mask = hv_arr[:, 1] <= HV_CUTOFF
                wls, mdeg = wls[mask], mdeg[mask]

        if len(wls) < 15:
            continue

        s_idx = np.argsort(wls)
        wls, mdeg = wls[s_idx], mdeg[s_idx]

        if len(mdeg) >= 11:
            mdeg = savgol_filter(mdeg, 15, 3)

        mre = mdeg_to_mre(mdeg)
        spectra_by_conc[float(conc)] = (wls, mre)

    concs = np.array(sorted(spectra_by_conc.keys()))
    y_217 = np.array([np.interp(217.0, spectra_by_conc[c][0], spectra_by_conc[c][1]) for c in concs])

    return spectra_by_conc, concs, y_217


def load_fluo_data(folder_path):
    conc_map = load_concentrations(folder_path)
    spectra_by_conc = {}
    csm_results = []
    ratio_results = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".csv"):
            continue
        matches = re.findall(r"(\d+)", fname)
        if not matches:
            continue
        sn = int(matches[-1])
        conc = conc_map.get(sn, None)
        if conc is None:
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath, header=1, usecols=[0, 1])
            df.columns = ["Wavelength", "Intensity"]
            df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
            df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
            df.dropna(inplace=True)
            if df.empty:
                continue

            step = df["Wavelength"].diff().median()
            w_pts = max(3, int(np.ceil(15.0 / step))) if step > 0 else 11
            w_pts = w_pts + 1 if w_pts % 2 == 0 else w_pts

            if len(df) >= w_pts:
                df["Intensity"] = savgol_filter(df["Intensity"], w_pts, 3)

            wls = df["Wavelength"].values
            intens_raw = df["Intensity"].values.copy()

            # 1. Ratio I310 / I332 liczone BEZ odejmowania baseline
            i310 = np.interp(310.0, wls, intens_raw)
            i332 = np.interp(332.0, wls, intens_raw)
            ratio_val = (i310 / i332) if i332 != 0 else np.nan
            ratio_results.append((float(conc), ratio_val))

            # 2. Korekta baseline na 400 nm dla widm i analizy CSM/AEW
            b_idx = (df["Wavelength"] - 400.0).abs().idxmin()
            intens_corr = intens_raw - intens_raw[b_idx]
            spectra_by_conc[float(conc)] = (wls, intens_corr)

            csm_mask = (wls >= CSM_MIN_WL) & (wls <= CSM_MAX_WL)
            w_csm = wls[csm_mask]
            i_csm = np.clip(intens_corr[csm_mask], 0, None)
            tot_i = i_csm.sum()
            csm_val = (w_csm * i_csm).sum() / tot_i if tot_i > 0 else np.nan
            csm_results.append((float(conc), csm_val))

        except Exception:
            continue

    df_csm = pd.DataFrame(csm_results, columns=["conc", "val"]).dropna().sort_values("conc")
    df_ratio = pd.DataFrame(ratio_results, columns=["conc", "val"]).dropna().sort_values("conc")

    return spectra_by_conc, df_csm["conc"].values, df_csm["val"].values, df_ratio["conc"].values, df_ratio["val"].values


# ==============================================================================
# 4. Generator Figury 4.7 (Siatka 2 x 3)
# ==============================================================================
def generate_figure_4_7():
    print("Wczytywanie i przetwarzanie danych...")
    cd_spectra, concs_cd, y_cd_217 = load_cd_prodata(CD_DATA_DIR)
    fluo_spectra, concs_fl, y_fl_csm, concs_ratio, y_ratio = load_fluo_data(FLUO_DATA_DIR)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True, dpi=300)
    ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2]

    # --------------------------------------------------------------------------
    # PANEL A: CD 217 nm Unfolding (2-state vs 3-state fit)
    # --------------------------------------------------------------------------
    ax_a.scatter(
        concs_cd, y_cd_217,
        color=COLOR_FS, edgecolor=COLOR_FS_DARK,
        s=48, marker="D", linewidths=1.1, zorder=4,
        label=r"$Fs$TrmD–Tm1570 (217 nm)"
    )

    x_dense_cd = np.linspace(concs_cd.min(), concs_cd.max(), 300)

    # Model 2-stanowy
    p0_2s = [y_cd_217[0], y_cd_217[-1], 0.94, 2.91]
    try:
        popt_2s, pcov_2s = curve_fit(two_state_model, concs_cd, y_cd_217, p0=p0_2s, maxfev=10000)
        perr_2s = np.sqrt(np.diag(pcov_2s))
        r2_2s = calc_r_squared(y_cd_217, two_state_model(concs_cd, *popt_2s))
        ax_a.plot(
            x_dense_cd, two_state_model(x_dense_cd, *popt_2s),
            color=COLOR_2STATE, ls="--", lw=1.6, zorder=2,
            label=f"2-state fit ($R^2={r2_2s:.3f}$)"
        )
        export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm_2state_comp", TIME_H, CONC_UM, "2-state",
                       popt_2s[3], perr_2s[3], popt_2s[2], perr_2s[2], r2_2s)
    except Exception as e:
        print(f"Błąd fitu 2-state CD: {e}")

    # Model 3-stanowy
    y_min_cd, y_max_cd = np.min(y_cd_217), np.max(y_cd_217)
    a_n_cd = y_cd_217[0]
    a_u_cd = y_cd_217[-1]
    a_i_cd = a_n_cd + 0.45 * (a_u_cd - a_n_cd)

    p0_3s = [a_n_cd, a_i_cd, a_u_cd, 3.98, 2.34, 0.93, 3.68]
    bounds_3s = (
        [y_min_cd - 2000, y_min_cd - 2000, y_min_cd - 2000, 1.5, 2.00, 0.3, 3.20],
        [y_max_cd + 2000, y_max_cd + 2000, y_max_cd + 2000, 7.0, 2.65, 2.5, 4.20]
    )
    try:
        popt_3s, pcov_3s = curve_fit(three_state_model, concs_cd, y_cd_217, p0=p0_3s, bounds=bounds_3s, maxfev=30000)
        perr_3s = np.sqrt(np.diag(pcov_3s))
        r2_3s = calc_r_squared(y_cd_217, three_state_model(concs_cd, *popt_3s))

        ax_a.plot(
            x_dense_cd, three_state_model(x_dense_cd, *popt_3s),
            color=COLOR_FS_DARK, ls="-", lw=2.2, zorder=3,
            label=f"3-state fit ($R^2={r2_3s:.3f}$)"
        )
        export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm_Step1", TIME_H, CONC_UM, "3-state_trans1",
                       popt_3s[4], perr_3s[4], popt_3s[3], perr_3s[3], r2_3s)
        export_fit_row(FIGURE_ID, CONSTRUCT, "CD_217nm_Step2", TIME_H, CONC_UM, "3-state_trans2",
                       popt_3s[6], perr_3s[6], popt_3s[5], perr_3s[5], r2_3s)
    except Exception as e:
        print(f"Błąd fitu 3-state CD: {e}")

    ax_a.set_xlim(-0.2, max(concs_cd) + 0.3)
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=11)
    ax_a.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=11)
    ax_a.tick_params(labelsize=10)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower right", bbox_to_anchor=(0.96, 0.05), fontsize=9, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_a.text(0.04, 0.94, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_a.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 | CD 217 nm", fontweight="bold", fontsize=11.5, pad=8)

    # --------------------------------------------------------------------------
    # PANEL B: Fluorescence AEW <lambda> (2-state fit + plateau)
    # --------------------------------------------------------------------------
    ax_b.scatter(
        concs_fl, y_fl_csm,
        color=COLOR_FS, edgecolor=COLOR_FS_DARK,
        s=48, marker="D", linewidths=1.1, zorder=4,
        label=r"AEW $\langle \lambda \rangle$"
    )

    x_dense_fl = np.linspace(concs_fl.min(), concs_fl.max(), 300)
    p0_fl_2s = [y_fl_csm[0], y_fl_csm[-1], 1.16, 1.91]
    try:
        popt_fl_2s, pcov_fl_2s = curve_fit(two_state_model, concs_fl, y_fl_csm, p0=p0_fl_2s, maxfev=10000)
        perr_fl_2s = np.sqrt(np.diag(pcov_fl_2s))
        r2_fl_2s = calc_r_squared(y_fl_csm, two_state_model(concs_fl, *popt_fl_2s))

        ax_b.plot(
            x_dense_fl, two_state_model(x_dense_fl, *popt_fl_2s),
            color=COLOR_FS_DARK, ls="-", lw=2.0, zorder=3,
            label=f"2-state fit ($R^2={r2_fl_2s:.3f}$)"
        )
        export_fit_row(FIGURE_ID, CONSTRUCT, "Fluorescence_CSM", TIME_H, CONC_UM, "2-state",
                       popt_fl_2s[3], perr_fl_2s[3], popt_fl_2s[2], perr_fl_2s[2], r2_fl_2s)
    except Exception as e:
        print(f"Błąd fitu Fluo CSM: {e}")

    ax_b.set_xlim(-0.2, max(concs_fl) + 0.3)
    ax_b.set_xlabel("[GdmCl] (M)", fontsize=11)
    ax_b.set_ylabel(r"Average Emission Wavelength $\langle \lambda \rangle$ (nm)", fontsize=11)
    ax_b.tick_params(labelsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="lower right", bbox_to_anchor=(0.96, 0.05), fontsize=9, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_b.text(0.04, 0.94, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_b.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 | Fluorescence AEW $\boldsymbol{\langle \lambda \rangle}$",
                   fontweight="bold", fontsize=11.5, pad=8)
    # --------------------------------------------------------------------------
    # PANEL C: Fluorescence Ratio I310 / I332 (przeniesiony z F)
    # --------------------------------------------------------------------------
    ax_c.scatter(
        concs_ratio, y_ratio,
        color=COLOR_RATIO, edgecolor="#3F3B6C",
        s=48, marker="D", linewidths=1.1, zorder=4,
        label=r"$I_{310} / I_{332}$"
    )

    x_dense_rat = np.linspace(concs_ratio.min(), concs_ratio.max(), 300)
    p0_rat = [y_ratio[0], y_ratio[-1], 0.73, 2.17]
    try:
        popt_rat, pcov_rat = curve_fit(two_state_model, concs_ratio, y_ratio, p0=p0_rat, maxfev=10000)
        perr_rat = np.sqrt(np.diag(pcov_rat))
        r2_rat = calc_r_squared(y_ratio, two_state_model(concs_ratio, *popt_rat))

        ax_c.plot(
            x_dense_rat, two_state_model(x_dense_rat, *popt_rat),
            color=COLOR_RATIO, ls="-", lw=2.0, zorder=3,
            label=f"2-state ($R^2={r2_rat:.3f}$)"
        )
        export_fit_row(FIGURE_ID, CONSTRUCT, "Fluorescence_Ratio_310_332", TIME_H, CONC_UM, "2-state",
                       popt_rat[3], perr_rat[3], popt_rat[2], perr_rat[2], r2_rat)
    except Exception as e:
        print(f"Błąd fitu Fluorescence Ratio: {e}")

    ax_c.set_xlim(-0.2, max(concs_ratio) + 0.3)
    ax_c.set_xlabel("[GdmCl] (M)", fontsize=11)
    ax_c.set_ylabel(r"Fluorescence Ratio $I_{310} / I_{332}$", fontsize=11)
    ax_c.tick_params(labelsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="lower right", bbox_to_anchor=(0.96, 0.05), fontsize=9, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_c.text(0.04, 0.94, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_c.set_title(
        r"$\mathsf{Fs}$TrmD–Tm1570 | Fluorescence Ratio $\boldsymbol{I}_{\mathbf{310}} \boldsymbol{/} \boldsymbol{I}_{\mathbf{332}}$",
        fontweight="bold", fontsize=11.5, pad=8)
    # --------------------------------------------------------------------------
    # PANEL D: Far-UV CD Spectra (210-250 nm, plasma)
    # --------------------------------------------------------------------------
    norm_cd = Normalize(vmin=concs_cd.min(), vmax=concs_cd.max())
    for c in concs_cd:
        wl, mre = cd_spectra[c]
        mask = (wl >= 210.0) & (wl <= 250.0)
        ax_d.plot(wl[mask], mre[mask], color=plasma_trimmed(norm_cd(c)), lw=1.35)

    ax_d.set_xlim(210.0, 250.0)
    ax_d.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_d.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=11)
    ax_d.tick_params(labelsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.text(0.04, 0.94, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_d.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 – Far-UV CD Spectra", fontweight="bold", fontsize=11.5, pad=8)

    cax_d = ax_d.inset_axes([0.82, 0.08, 0.035, 0.38])
    sm_d = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd)
    sm_d.set_array([])
    cbar_d = fig.colorbar(sm_d, cax=cax_d, orientation="vertical")
    cbar_d.ax.tick_params(labelsize=8)
    cbar_d.set_label("[GdmCl] (M)", fontsize=8, labelpad=4)

    # --------------------------------------------------------------------------
    # PANEL E: Fluorescence Emission Spectra (310-400 nm, plasma)
    # --------------------------------------------------------------------------
    norm_fl = Normalize(vmin=concs_fl.min(), vmax=concs_fl.max())
    for c in concs_fl:
        wl, inten = fluo_spectra[c]
        mask = (wl >= 310.0) & (wl <= 400.0)
        ax_e.plot(wl[mask], inten[mask], color=plasma_trimmed(norm_fl(c)), lw=1.35)

    ax_e.set_xlim(310.0, 400.0)
    ax_e.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_e.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=11)
    ax_e.tick_params(labelsize=10)
    ax_e.grid(True, linestyle=":", alpha=0.5)
    ax_e.text(0.04, 0.94, "E", transform=ax_e.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_e.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 – Fluorescence Emission Spectra", fontweight="bold", fontsize=11.5, pad=8)

    cax_e = ax_e.inset_axes([0.82, 0.54, 0.035, 0.38])
    sm_e = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fl)
    sm_e.set_array([])
    cbar_e = fig.colorbar(sm_e, cax=cax_e, orientation="vertical")
    cbar_e.ax.tick_params(labelsize=8)
    cbar_e.set_label("[GdmCl] (M)", fontsize=8, labelpad=4)

    # --------------------------------------------------------------------------
    # PANEL F: K2D Deconvolution (przeniesiony z C)
    # --------------------------------------------------------------------------
    alpha_list, beta_list, coil_list = [], [], []
    for c in concs_cd:
        w_i, mre_i = cd_spectra[c]
        a_f, b_f, c_f = estimate_k2d_secondary_structure(w_i, mre_i)
        alpha_list.append(a_f * 100.0)
        beta_list.append(b_f * 100.0)
        coil_list.append(c_f * 100.0)

    alpha_arr = np.array(alpha_list)
    beta_arr = np.array(beta_list)
    coil_arr = np.array(coil_list)

    ax_f.scatter(concs_cd, alpha_arr, color=COLOR_ALPHA, marker="o", s=45, label=r"$\alpha$-helix", zorder=3)
    ax_f.plot(concs_cd, savgol_filter(alpha_arr, min(7, len(alpha_arr)), 2), color=COLOR_ALPHA, lw=1.8, ls="-")

    ax_f.scatter(concs_cd, beta_arr, color=COLOR_BETA, marker="s", s=42, label=r"$\beta$-sheet", zorder=3)
    ax_f.plot(concs_cd, savgol_filter(beta_arr, min(7, len(beta_arr)), 2), color=COLOR_BETA, lw=1.8, ls="-")

    ax_f.scatter(concs_cd, coil_arr, color=COLOR_COIL, marker="^", s=45, label=r"Random coil", zorder=3)
    ax_f.plot(concs_cd, savgol_filter(coil_arr, min(7, len(coil_arr)), 2), color=COLOR_COIL, lw=1.8, ls="-")

    ax_f.set_xlim(-0.2, max(concs_cd) + 0.3)
    ax_f.set_ylim(-5, 105)
    ax_f.set_xlabel("[GdmCl] (M)", fontsize=11)
    ax_f.set_ylabel("Secondary Structure Content (%)", fontsize=11)
    ax_f.tick_params(labelsize=10)
    ax_f.grid(True, linestyle=":", alpha=0.5)
    ax_f.legend(loc="center right", bbox_to_anchor=(0.96, 0.50), fontsize=9, frameon=True,
                facecolor="white", framealpha=0.9, edgecolor="none", markerfirst=False)
    ax_f.text(0.04, 0.94, "F", transform=ax_f.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_f.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 | K2D Deconvolution", fontweight="bold", fontsize=11.5, pad=8)

    # --------------------------------------------------------------------------
    # Zapis plików publikacyjnych
    # --------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"\n[✓] Gotowa Figura 4.7 (2x3) zapisana do:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")
    print(f"[✓] Parametry dopasowań zaktualizowano w: {CSV_PATH}")


if __name__ == "__main__":
    generate_figure_4_7()